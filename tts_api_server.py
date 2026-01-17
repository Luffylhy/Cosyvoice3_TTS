# -*- coding: utf-8 -*-
"""
TTS API服务器 - 支持多语言、多情绪、多性别的实时流式语音合成
支持：英语、俄语、法语、中文
情绪：专业、亲切、兴奋
性别：男声、女声

优化策略：
1. 音频预缓冲机制 - 在RTF波动时提供平稳的流式输出
2. 进程优先级提升 - 减少被其他服务抢占的概率
3. GPU资源优化 - 减少显存碎片和CUDA overhead
"""
import os
import sys
import argparse
import logging
from typing import Optional
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchaudio
import threading
import queue
import time

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav


# ==================== 性能优化配置 ====================

def set_process_priority():
    """设置进程优先级，减少被其他服务抢占的概率"""
    try:
        import platform
        if platform.system() == 'Windows':
            import ctypes
            # 设置为高优先级（不是实时优先级，避免系统不稳定）
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 
                0x00008000  # ABOVE_NORMAL_PRIORITY_CLASS
            )
            logger.info("已设置Windows进程优先级为 ABOVE_NORMAL")
        else:
            # Linux/Mac: 使用nice值降低（需要适当权限）
            try:
                os.nice(-5)  # 提高优先级
                logger.info("已设置进程nice值为-5")
            except PermissionError:
                logger.warning("无权限设置进程优先级，跳过")
    except Exception as e:
        logger.warning(f"设置进程优先级失败: {e}")


def optimize_torch_settings():
    """优化PyTorch设置以提高推理性能"""
    # 启用cuDNN benchmark，对于固定大小的输入可以加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # 使用确定性算法可能更稳定，但benchmark在大多数情况下更快
        # torch.backends.cudnn.deterministic = True
        
        # 设置默认的CUDA内存分配策略
        # 使用expandable_segments可以减少内存碎片
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            logger.info("已启用CUDA expandable_segments内存分配策略")
        except Exception as e:
            logger.debug(f"CUDA内存分配策略设置失败: {e}")
        
        # 限制CUDA内存预留，避免与其他服务竞争
        # torch.cuda.set_per_process_memory_fraction(0.8)  # 可选：限制为80%显存
        
        logger.info(f"CUDA设备: {torch.cuda.get_device_name()}")
        logger.info(f"cuDNN benchmark已启用")


class AudioBuffer:
    """
    音频预缓冲区 - 用于平滑RTF波动
    
    工作原理：
    1. 生产者线程生成音频块并放入缓冲区
    2. 消费者按固定间隔从缓冲区取出音频块
    3. 当RTF暂时>1时，可以从缓冲区中取出之前缓存的音频
    4. 当RTF<1时，缓冲区会逐渐填满
    """
    
    def __init__(self, 
                 min_buffer_chunks: int = 2,    # 最小缓冲块数（开始输出前需要的块数）
                 max_buffer_chunks: int = 10,   # 最大缓冲块数
                 timeout: float = 0.1):         # 获取超时时间
        self.buffer = queue.Queue(maxsize=max_buffer_chunks)
        self.min_buffer_chunks = min_buffer_chunks
        self.max_buffer_chunks = max_buffer_chunks
        self.timeout = timeout
        self.is_finished = False
        self.error = None
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        
    def put(self, audio_bytes: bytes):
        """将音频块放入缓冲区"""
        try:
            self.buffer.put(audio_bytes, timeout=self.timeout * 10)
            # 当缓冲区达到最小要求时，通知消费者可以开始
            if self.buffer.qsize() >= self.min_buffer_chunks:
                self._ready_event.set()
        except queue.Full:
            logger.warning("音频缓冲区已满，等待消费者处理")
            self.buffer.put(audio_bytes)  # 阻塞等待
    
    def get(self) -> Optional[bytes]:
        """从缓冲区获取音频块"""
        try:
            # 首次获取时等待缓冲区填充到最小要求
            if not self._ready_event.is_set():
                self._ready_event.wait(timeout=self.timeout * self.min_buffer_chunks * 5)
            
            return self.buffer.get(timeout=self.timeout)
        except queue.Empty:
            if self.is_finished:
                return None
            # 超时但未结束，返回None让调用者决定如何处理
            return None
    
    def finish(self, error: Optional[Exception] = None):
        """标记生产完成"""
        with self._lock:
            self.is_finished = True
            self.error = error
            self._ready_event.set()  # 确保消费者不会永久等待
    
    def has_data(self) -> bool:
        """检查缓冲区是否有数据"""
        return not self.buffer.empty()
    
    def is_done(self) -> bool:
        """检查是否已完成（无更多数据且已标记完成）"""
        return self.is_finished and self.buffer.empty()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局模型实例
cosyvoice_model = None
# 模型加载标志，防止重复加载
_model_loaded = False
# 优化配置
_optimization_applied = False


def apply_optimizations():
    """应用全局优化设置"""
    global _optimization_applied
    if _optimization_applied:
        return
    
    logger.info("正在应用性能优化设置...")
    
    # 1. 设置进程优先级
    set_process_priority()
    
    # 2. 优化PyTorch设置
    optimize_torch_settings()
    
    # 3. 设置线程数优化（在多服务环境下减少CPU竞争）
    # 根据CPU核心数设置合理的线程数
    try:
        cpu_count = os.cpu_count() or 4
        # 在多服务环境下，限制线程数以减少竞争
        optimal_threads = max(2, cpu_count // 2)
        torch.set_num_threads(optimal_threads)
        logger.info(f"PyTorch线程数设置为: {optimal_threads}")
    except Exception as e:
        logger.debug(f"设置线程数失败: {e}")
    
    _optimization_applied = True
    logger.info("性能优化设置完成")


def init_model(model_dir: str, load_vllm: bool = False, fp16: bool = False, 
               flow_timesteps: int = 10, token_hop_len: int = 25):
    """初始化TTS模型
    
    Args:
        model_dir: 模型目录路径
        load_vllm: 是否使用vLLM加速（默认: False）
        fp16: 是否使用FP16精度（默认: False）
        flow_timesteps: Flow diffusion步数（默认: 10）
            - 10: 高质量
            - 8: 平衡模式，RTF降低约20%
            - 6: 低延迟模式，RTF降低约40%
        token_hop_len: 流式处理token跳跃长度（默认: 25）
            - 25: 默认，首包延迟约0.5s
            - 35: 平衡模式，RTF降低约15%
            - 50: 低RTF模式，RTF降低约25%
    """
    global cosyvoice_model, _model_loaded
    
    # 应用优化设置
    apply_optimizations()
    
    if _model_loaded and cosyvoice_model is not None:
        logger.info("模型已加载，跳过重复加载")
        # 如果模型已加载，仍然更新参数
        if hasattr(cosyvoice_model.model, 'set_flow_timesteps'):
            cosyvoice_model.model.set_flow_timesteps(flow_timesteps)
            logger.info(f"已更新flow_timesteps为: {flow_timesteps}")
        if hasattr(cosyvoice_model.model, 'set_token_hop_len'):
            cosyvoice_model.model.set_token_hop_len(token_hop_len)
            logger.info(f"已更新token_hop_len为: {token_hop_len}")
        return
    try:
        logger.info(f"正在加载TTS模型: {model_dir}")
        if load_vllm:
            logger.info("启用vLLM加速模式")
        if fp16:
            logger.info("启用FP16精度模式")
        logger.info(f"Flow diffusion步数: {flow_timesteps}")
        logger.info(f"Token跳跃长度: {token_hop_len}")
        
        # 使用FP16可以减少显存使用并提高推理速度
        cosyvoice_model = AutoModel(model_dir=model_dir, load_vllm=load_vllm, fp16=fp16)
        
        # 设置flow_timesteps
        if hasattr(cosyvoice_model.model, 'set_flow_timesteps'):
            cosyvoice_model.model.set_flow_timesteps(flow_timesteps)
        elif hasattr(cosyvoice_model.model, 'flow_n_timesteps'):
            cosyvoice_model.model.flow_n_timesteps = flow_timesteps
        
        # 设置token_hop_len
        if hasattr(cosyvoice_model.model, 'set_token_hop_len'):
            cosyvoice_model.model.set_token_hop_len(token_hop_len)
        elif hasattr(cosyvoice_model.model, 'token_hop_len'):
            cosyvoice_model.model.token_hop_len = token_hop_len
        
        _model_loaded = True
        logger.info("TTS模型加载成功")
        
        # 打印显存使用情况和优化配置
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存使用: 已分配 {allocated:.2f}GB, 已预留 {reserved:.2f}GB")
        
        logger.info(f"RTF优化配置: flow_timesteps={flow_timesteps}, token_hop_len={token_hop_len}")
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型（如果还未加载）
    global cosyvoice_model, _model_loaded
    
    if not _model_loaded:
        # 模型路径从环境变量获取（如果通过命令行参数传入，会在main函数中处理）
        model_dir = os.getenv("MODEL_DIR")
        load_vllm = os.getenv("LOAD_VLLM", "false").lower() in ("true", "1", "yes")
        fp16 = os.getenv("FP16", "false").lower() in ("true", "1", "yes")
        
        if model_dir:
            if os.path.exists(model_dir):
                init_model(model_dir, load_vllm=load_vllm, fp16=fp16)
            else:
                # 可能是ModelScope/HuggingFace模型ID，让AutoModel自动下载
                logger.info(f"模型目录不存在，将尝试从ModelScope/HuggingFace下载: {model_dir}")
                init_model(model_dir, load_vllm=load_vllm, fp16=fp16)
        else:
            # 检查默认的models文件夹
            default_model_dir = os.path.join(ROOT_DIR, "models", "Fun-CosyVoice3-0.5B-2512")
            if os.path.exists(default_model_dir):
                logger.info(f"使用默认模型目录: {default_model_dir}")
                init_model(default_model_dir, load_vllm=load_vllm, fp16=fp16)
            else:
                logger.info("未设置MODEL_DIR环境变量，模型将在首次请求时加载（如果通过--model_dir参数指定）")
    
    yield
    
    # 关闭时清理（如果需要）
    # 目前不需要特殊清理


app = FastAPI(title="CosyVoice TTS API", version="1.0.0", lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 语言映射
LANGUAGE_MAP = {
    "en": "english",
    "ru": "russian", 
    "fr": "french",
    "zh": "chinese",
    "english": "en",
    "russian": "ru",
    "french": "fr",
    "chinese": "zh"
}

# 情绪映射到CosyVoice支持的情绪标记
# 专业 -> NEUTRAL, 亲切 -> HAPPY (温和), 兴奋 -> HAPPY (强烈)
EMOTION_MAP = {
    "professional": "NEUTRAL",
    "专业": "NEUTRAL",
    "friendly": "HAPPY",
    "亲切": "HAPPY",
    "excited": "HAPPY",
    "兴奋": "HAPPY"
}

# 音色配置
# 格式：{性别: {音色ID: {语言: 风格描述}}}
# 注意：instruct文本需要简洁，模型才能正确识别为控制指令而非要合成的文本
# 文件命名格式：male1.wav, male2.wav, male3.wav, female1.wav, female2.wav, female3.wav
VOICE_CONFIGS = {
    "male": {
        "1": {
            "name": "孙悟空",
            "zh": "请模仿王者荣耀中孙悟空的语音风格：粗犷沙哑的男性声线，自信张扬、狂放不羁，语速稍快，节奏感强，带轻微的西北方言腔调。",
            "en": "Please imitate Sun Wukong's voice style from Honor of Kings: rough and hoarse male voice, confident and flamboyant, slightly fast pace with strong rhythm, slight northwestern dialect accent.",
            "ru": "Пожалуйста, имитируйте голосовой стиль Сунь Укуна из Honor of Kings: грубый и хриплый мужской голос, уверенный и яркий, слегка быстрый темп с сильным ритмом, легкий северо-западный диалектный акцент.",
            "fr": "Veuillez imiter le style vocal de Sun Wukong d'Honor of Kings: voix masculine rugueuse et rauque, confiante et flamboyante, rythme légèrement rapide avec rythme fort, léger accent dialectal du nord-ouest."
        },
        "2": {
            "name": "猪八戒",
            "zh": "请模仿王者荣耀中猪八戒的语音风格：憨厚慵懒男声，软糯随性，憨萌贪吃，语速慢，带哼哼语气词，亲切接地气。",
            "en": "Please imitate Zhu Bajie's voice style from Honor of Kings: honest lazy male voice, soft and casual, naive and greedy, slow pace, with snorting particles, approachable.",
            "ru": "Пожалуйста, имитируйте голосовой стиль Чжу Бацзе из Honor of Kings: простой ленивый мужской голос, мягкий непринужденный, наивный прожорливый, медленный темп, с храпящими частицами, доступный.",
            "fr": "Veuillez imiter le style vocal de Zhu Bajie d'Honor of Kings: voix masculine honnête et paresseuse, douce et décontractée, naïve et gourmande, rythme lent, avec particules de reniflement, accessible."
        },
        "3": {
            "name": "太乙真人",
            "zh": "请用四川话表达，完全按照参考音频的说话风格和语调进行模仿。参考音频是地道四川方言男声，浓重川音口音，高亮诙谐，语速快嗓门大，带俏皮口头禅，咬字带川味。",
            "en": "Please imitate Taiyi Zhenren's voice style from Nezha 2: male voice with Sichuan accent, bright and humorous, wine-loving and casual, fast pace & loud voice, with playful catchphrases.",
            "ru": "Пожалуйста, имитируйте голосовой стиль Тайи Чжэньжэня из Нечжа 2: мужской голос с сичуанским акцентом, яркий юмористический, любящий вино непринужденный, быстрый темп и громкий голос, с игривыми слоганами.",
            "fr": "Veuillez imiter le style vocal de Taiyi Zhenren de Nezha 2: voix masculine avec accent du Sichuan, brillante et humoristique, amatrice de vin et décontractée, rythme rapide et voix forte, avec mots d'ordre amusants."
        }
    },
    "female": {
        "1": {
            "name": "武则天",
            "zh": "请模仿王者荣耀中武则天的语音风格：高贵冷艳的女性声线，声调偏低沉，雍容华贵、不怒自威，语速缓慢且沉稳，清晰优雅，带帝王气场。",
            "en": "Please imitate Wu Zetian's voice style from Honor of Kings: noble and cold female voice, slightly low-pitched, graceful and majestic, slow and steady pace, clear and elegant, with imperial aura.",
            "ru": "Пожалуйста, имитируйте голосовой стиль У Цзэтянь из Honor of Kings: благородный и холодный женский голос, слегка низкий, грациозный и величественный, медленный и устойчивый темп, четкий и элегантный, с императорской аурой.",
            "fr": "Veuillez imiter le style vocal de Wu Zetian d'Honor of Kings: voix féminine noble et froide, légèrement grave, gracieuse et majestueuse, rythme lent et stable, claire et élégante, avec aura impériale."
        },
        "2": {
            "name": "林志玲",
            "zh": "请用台湾话表达，完全按照参考音频的说话风格和语调进行模仿。参考音频是甜美娇柔女声，台湾口音明显，高柔舒缓，嗲尾音，优雅温柔，亲和力拉满，咬字带台味。",
            "en": "Please imitate Lin Chi-ling's voice style: sweet delicate female voice, high and soft, slow pace, coquettish tail sound, elegant tender, full of approachability.",
            "ru": "Пожалуйста, имитируйте голосовой стиль Линь Чжилин: сладкий деликатный женский голос, высокий мягкий, медленный темп, кокетливый завершающий звук, элегантный нежный, полный доступности.",
            "fr": "Veuillez imiter le style vocal de Lin Chi-ling: voix féminine douce et délicate, haute et douce, rythme lent, son final coquet, élégante et tendre, pleine d'accessibilité."
        },
        "3": {
            "name": "东北雨姐",
            "zh": "请用东北话表达，完全按照参考音频的说话风格和语调进行模仿，浓重东北口音，豪爽热烈，语速快嗓门大，粗粝沙哑，真诚接地气，带直来直去的烟火气，咬字带东北味。",
            "en": "Please imitate Northeast Rain Sister's voice style: authentic Northeast Chinese dialect female voice, bold and warm, fast pace & loud voice, gritty and hoarse, sincere and down-to-earth, with straightforward earthiness.",
            "ru": "Пожалуйста, имитируйте голосовой стиль Северо-Восточной Сестры Дождя: подлинный женский голос с северо-восточным китайским диалектом, смелый и теплый, быстрый темп и громкий голос, грубый и хриплый, искренний и приземленный, с прямолинейной земной атмосферой.",
            "fr": "Veuillez imiter le style vocal de Sœur Pluie du Nord-Est: voix féminine avec dialecte du Nord-Est de la Chine authentique, audacieuse et chaleureuse, rythme rapide et voix forte, rêche et enrouée, sincère et terre-à-terre, avec une atmosphère terre-à-terre directe."
        }
    }
}

# 向后兼容：保留旧的GENDER_PROMPTS结构（使用1作为默认）
GENDER_PROMPTS = {
    "male": VOICE_CONFIGS["male"]["1"],
    "female": VOICE_CONFIGS["female"]["1"],
    "男": VOICE_CONFIGS["male"]["1"],
    "女": VOICE_CONFIGS["female"]["1"]
}

# 情绪微调描述（作为音色风格的补充，而非覆盖）
# 注意：这些描述会在保持音色特色的基础上微调情绪，不会覆盖音色的核心特征
EMOTION_ADJUSTMENTS = {
    "zh": {
        "professional": "在保持角色风格的基础上，语调更加专业和正式。",
        "专业": "在保持角色风格的基础上，语调更加专业和正式。",
        "friendly": "在保持角色风格的基础上，语调更加友好和亲切。",
        "亲切": "在保持角色风格的基础上，语调更加友好和亲切。",
        "excited": "在保持角色风格的基础上，语调更加兴奋和充满活力。",
        "兴奋": "在保持角色风格的基础上，语调更加兴奋和充满活力。"
    },
    "en": {
        "professional": "While maintaining the character's style, speak more professionally and formally.",
        "专业": "While maintaining the character's style, speak more professionally and formally.",
        "friendly": "While maintaining the character's style, speak more friendly and approachable.",
        "亲切": "While maintaining the character's style, speak more friendly and approachable.",
        "excited": "While maintaining the character's style, speak more excitedly and energetically.",
        "兴奋": "While maintaining the character's style, speak more excitedly and energetically."
    },
    "ru": {
        "professional": "Сохраняя стиль персонажа, говорите более профессионально и формально.",
        "专业": "Сохраняя стиль персонажа, говорите более профессионально и формально.",
        "friendly": "Сохраняя стиль персонажа, говорите более дружелюбно и доступно.",
        "亲切": "Сохраняя стиль персонажа, говорите более дружелюбно и доступно.",
        "excited": "Сохраняя стиль персонажа, говорите более взволнованно и энергично.",
        "兴奋": "Сохраняя стиль персонажа, говорите более взволнованно и энергично."
    },
    "fr": {
        "professional": "En maintenant le style du personnage, parlez de manière plus professionnelle et formelle.",
        "专业": "En maintenant le style du personnage, parlez de manière plus professionnelle et formelle.",
        "friendly": "En maintenant le style du personnage, parlez de manière plus amicale et accessible.",
        "亲切": "En maintenant le style du personnage, parlez de manière plus amicale et accessible.",
        "excited": "En maintenant le style du personnage, parlez de manière plus excitée et énergique.",
        "兴奋": "En maintenant le style du personnage, parlez de manière plus excitée et énergique."
    }
}


def build_instruct_text(language: str, emotion: str, gender: str, voice_id: str = "1", enable_emotion: bool = True) -> str:
    """
    构建instruct文本，用于控制语言和角色风格
    
    Args:
        language: 语言代码 (en/ru/fr/zh)
        emotion: 情绪（professional/friendly/excited），如果为None或空字符串则不添加情绪控制
        gender: 性别 (male/female)
        voice_id: 音色ID (1/2/3)
        enable_emotion: 是否启用情绪微调（默认: True）
    
    Returns:
        instruct文本
    """
    # 获取性别对应的音色配置
    gender_key = gender.lower()
    if gender_key in ["男", "male"]:
        gender_key = "male"
    elif gender_key in ["女", "female"]:
        gender_key = "female"
    else:
        gender_key = "male"  # 默认男声
    
    # 验证音色ID（支持字符串和数字格式）
    voice_id_str = str(voice_id).strip()
    if voice_id_str not in VOICE_CONFIGS[gender_key]:
        voice_id_str = "1"  # 默认使用1
        logger.warning(f"无效的音色ID {voice_id}，使用默认音色: 1")
    
    # 获取对应语言的角色风格描述
    voice_config = VOICE_CONFIGS[gender_key][voice_id_str]
    base_prompt = voice_config.get(language, voice_config["en"])
    
    # 添加情绪微调（如果启用且提供了有效情绪）
    emotion_adj = ""
    if enable_emotion and emotion:
        emotion_key = emotion.lower().strip()
        if emotion_key and emotion_key not in ["", "none", "无"]:
            emotion_adj = EMOTION_ADJUSTMENTS.get(language, {}).get(emotion_key, "")
            if emotion_adj:
                logger.debug(f"添加情绪微调: {emotion_key}")
    
    # 构建完整的instruct文本
    # 使用"You are a helpful assistant."作为基础，然后添加角色风格描述和情绪微调
    if emotion_adj:
        instruct_text = f"You are a helpful assistant. {base_prompt} {emotion_adj}<|endofprompt|>"
    else:
        instruct_text = f"You are a helpful assistant. {base_prompt}<|endofprompt|>"
    
    return instruct_text


def format_text_with_language(text: str, language: str) -> str:
    """
    为文本添加语言标记
    
    Args:
        text: 原始文本
        language: 语言代码 (en/ru/fr/zh)
    
    Returns:
        带语言标记的文本
    """
    lang_tag = f"<|{language}|>"
    if not text.startswith(lang_tag):
        text = f"{lang_tag}{text}"
    return text


def generate_audio_stream(model_output, target_sample_rate=None, source_sample_rate=None, use_buffer=True):
    """
    生成音频流，支持重采样和预缓冲
    
    Args:
        model_output: 模型输出生成器
        target_sample_rate: 目标采样率（如果为None，则使用源采样率）
        source_sample_rate: 源采样率（模型输出采样率）
        use_buffer: 是否使用预缓冲（在高负载时提供更平稳的输出）
    
    Yields:
        音频数据块 (bytes)
    """
    resampler = None
    if target_sample_rate and source_sample_rate and target_sample_rate != source_sample_rate:
        # 创建重采样器
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_sample_rate,
            new_freq=target_sample_rate
        )
        logger.info(f"音频重采样: {source_sample_rate}Hz -> {target_sample_rate}Hz")
    
    def process_chunk(chunk):
        """处理单个音频块"""
        if 'tts_speech' not in chunk:
            return None
        
        # 获取音频数据
        audio_tensor = chunk['tts_speech']
        
        # 如果需要进行重采样
        if resampler is not None:
            # 确保是2D tensor (1, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            # 重采样
            audio_tensor = resampler(audio_tensor)
            # 转换回numpy
            audio_data = audio_tensor.squeeze().numpy()
        else:
            # 直接转换为numpy
            audio_data = audio_tensor.numpy()
        
        # 确保是单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # 转换为int16
        audio_int16 = (audio_data * (2 ** 15)).astype(np.int16)
        # 转换为字节
        return audio_int16.tobytes()
    
    if use_buffer:
        # 使用预缓冲模式：先缓存几个块再开始输出
        audio_buffer = AudioBuffer(min_buffer_chunks=2, max_buffer_chunks=8)
        
        def producer():
            """生产者线程：生成音频并放入缓冲区"""
            try:
                for chunk in model_output:
                    audio_bytes = process_chunk(chunk)
                    if audio_bytes:
                        audio_buffer.put(audio_bytes)
                audio_buffer.finish()
            except Exception as e:
                logger.error(f"生成音频时出错: {e}")
                audio_buffer.finish(error=e)
        
        # 启动生产者线程
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        # 消费者：从缓冲区取出音频块
        try:
            while True:
                audio_bytes = audio_buffer.get()
                if audio_bytes is not None:
                    yield audio_bytes
                elif audio_buffer.is_done():
                    break
                # 如果超时但未完成，继续等待
                
            # 检查是否有错误
            if audio_buffer.error:
                raise audio_buffer.error
        finally:
            producer_thread.join(timeout=1.0)
    else:
        # 直接模式：不使用缓冲
        try:
            for chunk in model_output:
                audio_bytes = process_chunk(chunk)
                if audio_bytes:
                    yield audio_bytes
        except Exception as e:
            logger.error(f"生成音频流时出错: {e}")
            raise


def generate_audio_stream_optimized(model_output, target_sample_rate=None, source_sample_rate=None):
    """
    优化版音频流生成器 - 使用预缓冲机制
    
    与 generate_audio_stream 相比：
    1. 默认启用预缓冲，在RTF波动时提供更平稳的输出
    2. 减少首包延迟：只需要2个块就开始输出
    3. 在高负载时更加稳定
    """
    return generate_audio_stream(
        model_output, 
        target_sample_rate=target_sample_rate, 
        source_sample_rate=source_sample_rate,
        use_buffer=True
    )


@app.get("/")
async def root():
    """API根路径，返回服务信息"""
    return {
        "service": "CosyVoice TTS API",
        "version": "1.0.0",
        "supported_languages": ["en", "ru", "fr", "zh"],
        "supported_emotions": ["professional", "friendly", "excited"],
        "supported_genders": ["male", "female"],
        "supported_voices": {
            "male": list(VOICE_CONFIGS["male"].keys()),
            "female": list(VOICE_CONFIGS["female"].keys())
        },
        "endpoints": {
            "/tts": "POST - 文本转语音（流式）",
            "/health": "GET - 健康检查"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy", "model_loaded": True}


@app.get("/voices")
async def get_available_voices():
    """获取可用的音色列表"""
    result = {}
    for gender in ["male", "female"]:
        result[gender] = {}
        for voice_id, config in VOICE_CONFIGS[gender].items():
            result[gender][voice_id] = {
                "name": config.get("name", voice_id),
                "description": {
                    "zh": config.get("zh", ""),
                    "en": config.get("en", ""),
                    "ru": config.get("ru", ""),
                    "fr": config.get("fr", "")
                }
            }
    return result


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="要合成的文本"),
    language: str = Form("en", description="语言代码: en(英语), ru(俄语), fr(法语), zh(中文)"),
    emotion: str = Form("professional", description="情绪: professional(专业), friendly(亲切), excited(兴奋)"),
    gender: str = Form("male", description="性别: male(男声), female(女声)"),
    voice_id: str = Form("1", description="音色ID: 1, 2, 3（默认: 1）"),
    enable_emotion: Optional[bool] = Form(None, description="是否启用情绪微调（默认: None，如果指定了emotion则自动启用，否则不启用）"),
    sample_rate: Optional[int] = Form(44100, description="输出采样率（Hz），可选值: 16000, 22050, 24000, 44100, 48000（默认: 44100）"),
    stream: bool = Form(True, description="是否流式返回")
):
    """
    文本转语音API
    
    支持参数:
    - text: 要合成的文本（必需）
    - language: 语言代码，可选值: en, ru, fr, zh（默认: en）
    - emotion: 情绪（已废弃，保留用于兼容性）
    - gender: 性别，可选值: male(男声), female(女声)（默认: male）
    - voice_id: 音色ID，可选值: 1, 2, 3（默认: 1）
      - 男声: 1=孙悟空, 2=猪八戒, 3=太乙真人
      - 女声: 1=武则天, 2=林志玲, 3=东北雨姐
    - enable_emotion: 是否启用情绪微调（默认: None，自动判断）
      - None: 自动判断 - 如果emotion不是默认值（professional/专业），则自动启用；否则不启用
      - False: 完全使用音色风格，保持音色特色纯粹，忽略emotion参数
      - True: 在保持音色特色的基础上，根据emotion参数微调情绪表达
    - sample_rate: 输出采样率（Hz，默认: 44100）
      - 可选值: 16000, 22050, 24000, 44100, 48000
      - 如果指定，会对生成的音频进行重采样
    - stream: 是否流式返回（默认: True）
    
    注意：
    - 每个音色需要对应的参考音频文件（格式：{gender}{voice_id}.wav，如male1.wav, female2.wav）
    - 情绪控制作为音色风格的微调，不会覆盖音色的核心特征（音色、口音等）
    
    返回:
    - 流式音频数据 (audio/wav格式)
    """
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="TTS模型未加载，请检查服务器配置")
    
    # 验证语言参数
    language = language.lower()
    if language not in ["en", "ru", "fr", "zh"]:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}，支持的语言: en, ru, fr, zh")
    
    # 验证情绪参数
    emotion = emotion.lower()
    valid_emotions = ["professional", "专业", "friendly", "亲切", "excited", "兴奋"]
    if emotion not in valid_emotions:
        raise HTTPException(status_code=400, detail=f"不支持的情绪: {emotion}，支持的情绪: professional, friendly, excited")
    
    # 验证性别参数
    gender = gender.lower()
    valid_genders = ["male", "female", "男", "女"]
    if gender not in valid_genders:
        raise HTTPException(status_code=400, detail=f"不支持的性别: {gender}，支持的性别: male, female")
    
    # 验证音色ID参数（支持字符串和数字格式）
    gender_key = "male" if gender.lower() in ["male", "男"] else "female"
    valid_voice_ids = list(VOICE_CONFIGS[gender_key].keys())
    voice_id_str = str(voice_id).strip()
    if voice_id_str not in valid_voice_ids:
        raise HTTPException(
            status_code=400, 
            detail=f"无效的音色ID: {voice_id}，{gender_key}支持的音色: {', '.join(valid_voice_ids)}"
        )
    
    try:
        # 自动判断是否启用情绪控制
        # 如果 enable_emotion 为 None，则根据是否指定了非默认 emotion 来决定
        if enable_emotion is None:
            # 如果 emotion 不是默认值（professional/专业），则自动启用情绪控制
            emotion_lower = emotion.lower().strip() if emotion else ""
            # 默认情绪值：professional, 专业
            default_emotions = ["professional", "专业"]
            auto_enable_emotion = bool(emotion_lower and emotion_lower not in ["", "none", "无"] + default_emotions)
        else:
            auto_enable_emotion = enable_emotion
        
        # 构建instruct文本（传入voice_id和enable_emotion）
        instruct_text = build_instruct_text(language, emotion, gender, voice_id_str, enable_emotion=auto_enable_emotion)
        
        # 格式化文本（添加语言标记）
        formatted_text = format_text_with_language(text, language)
        
        logger.info(f"TTS请求 - 语言: {language}, 情绪: {emotion}, 性别: {gender}")
        logger.info(f"文本: {text[:50]}...")
        logger.info(f"Instruct: {instruct_text}")
        
        # 根据性别和音色ID查找对应的prompt文件
        # 文件命名格式：{gender}{voice_id}.wav（例如：male1.wav, female2.wav）
        gender_key = "male" if gender.lower() in ["male", "男"] else "female"
        prompt_filename = f"{gender_key}{voice_id_str}.wav"
        default_prompt_wav = os.path.join(ROOT_DIR, "asset", prompt_filename)
        
        # 如果找不到对应的文件，尝试其他备选方案
        if not os.path.exists(default_prompt_wav):
            # 尝试同性别的其他音色
            for alt_voice_id in valid_voice_ids:
                if alt_voice_id != voice_id_str:
                    alt_prompt = os.path.join(ROOT_DIR, "asset", f"{gender_key}{alt_voice_id}.wav")
                    if os.path.exists(alt_prompt):
                        default_prompt_wav = alt_prompt
                        logger.warning(f"未找到 {prompt_filename}，使用备选音色 {alt_prompt}")
                        break
            else:
                # 尝试另一个性别的文件作为备选
                alt_gender = "female" if gender_key == "male" else "male"
                alt_prompt = os.path.join(ROOT_DIR, "asset", f"{alt_gender}1.wav")
                if os.path.exists(alt_prompt):
                    default_prompt_wav = alt_prompt
                    logger.warning(f"未找到 {prompt_filename}，使用备选性别文件 {alt_prompt}")
                else:
                    # 尝试通用的prompt文件
                    fallback_paths = [
                        os.path.join(ROOT_DIR, "asset", "zero_shot_prompt.wav"),
                        os.path.join(ROOT_DIR, "asset", "cross_lingual_prompt.wav"),
                    ]
                    for path in fallback_paths:
                        if os.path.exists(path):
                            default_prompt_wav = path
                            logger.warning(f"使用通用prompt文件: {path}")
                            break
        
        if not os.path.exists(default_prompt_wav):
            logger.error(f"未找到prompt文件: {default_prompt_wav}")
            raise HTTPException(
                status_code=500, 
                detail=f"未找到prompt音频文件。请确保asset文件夹下有对应的音频文件（如 {prompt_filename}），或使用/tts/zero_shot端点并提供prompt_wav"
            )
        
        logger.info(f"使用prompt文件: {default_prompt_wav}")
        
        # 使用inference_instruct2进行推理（CosyVoice2/3支持）
        # 这会根据instruct_text控制语言、情绪和性别特征
        model_output = cosyvoice_model.inference_instruct2(
            formatted_text,
            instruct_text,
            default_prompt_wav,
            stream=stream
        )
        
        # 确定输出采样率（默认44100Hz）
        output_sample_rate = sample_rate if sample_rate is not None else 44100
        
        # 验证采样率
        valid_sample_rates = [16000, 22050, 24000, 44100, 48000]
        if output_sample_rate not in valid_sample_rates:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的采样率: {output_sample_rate}，支持的采样率: {', '.join(map(str, valid_sample_rates))}"
            )
        
        # 返回流式音频响应
        # 使用优化后的音频流生成器，启用预缓冲以应对RTF波动
        return StreamingResponse(
            generate_audio_stream_optimized(model_output, target_sample_rate=output_sample_rate, source_sample_rate=cosyvoice_model.sample_rate),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Sample-Rate": str(output_sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"TTS处理出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS处理失败: {str(e)}")


@app.post("/tts/zero_shot")
async def text_to_speech_zero_shot(
    text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form(..., description="参考文本"),
    prompt_wav: str = Form(..., description="参考音频文件路径（相对于项目根目录）"),
    language: str = Form("en", description="语言代码: en(英语), ru(俄语), fr(法语)"),
    emotion: str = Form("professional", description="情绪: professional(专业), friendly(亲切), excited(兴奋)"),
    stream: bool = Form(True, description="是否流式返回")
):
    """
    零样本文本转语音API（使用自定义参考音频）
    
    支持参数:
    - text: 要合成的文本（必需）
    - prompt_text: 参考文本（必需）
    - prompt_wav: 参考音频文件路径（必需）
    - language: 语言代码（默认: en）
    - emotion: 情绪（默认: professional）
    - stream: 是否流式返回（默认: True）
    """
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="TTS模型未加载")
    
    try:
        # 验证语言参数
        language = language.lower()
        if language not in ["en", "ru", "fr", "zh"]:
            raise HTTPException(status_code=400, detail=f"不支持的语言: {language}")
        
        # 构建完整的prompt文本（包含instruct）
        instruct_text = build_instruct_text(language, emotion, "male")  # 性别由prompt_wav决定
        full_prompt_text = f"{instruct_text} {prompt_text}"
        
        # 格式化文本
        formatted_text = format_text_with_language(text, language)
        
        # 加载prompt音频
        prompt_wav_path = os.path.join(ROOT_DIR, prompt_wav.lstrip("/"))
        if not os.path.exists(prompt_wav_path):
            raise HTTPException(status_code=404, detail=f"找不到prompt音频文件: {prompt_wav_path}")
        
        # 使用zero_shot推理
        model_output = cosyvoice_model.inference_zero_shot(
            formatted_text,
            full_prompt_text,
            prompt_wav_path,
            stream=stream
        )
        
        # 使用优化后的音频流生成器
        return StreamingResponse(
            generate_audio_stream_optimized(model_output, source_sample_rate=cosyvoice_model.sample_rate),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Sample-Rate": str(cosyvoice_model.sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"Zero-shot TTS处理出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Zero-shot TTS处理失败: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice TTS API服务器")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/Fun-CosyVoice3-0.5B-2512",
        help="模型目录路径或ModelScope/HuggingFace模型ID（默认: models/Fun-CosyVoice3-0.5B-2512）"
    )
    parser.add_argument(
        "--ssl_keyfile",
        type=str,
        default=None,
        help="SSL密钥文件路径（用于HTTPS）"
    )
    parser.add_argument(
        "--ssl_certfile",
        type=str,
        default=None,
        help="SSL证书文件路径（用于HTTPS）"
    )
    parser.add_argument(
        "--load_vllm",
        action="store_true",
        help="启用vLLM加速（需要模型支持）"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="启用FP16精度模式"
    )
    parser.add_argument(
        "--flow_timesteps",
        type=int,
        default=10,
        choices=[6, 8, 10],
        help="Flow diffusion步数：10(高质量), 8(平衡，RTF降低约20%%), 6(低延迟，RTF降低约40%%)"
    )
    parser.add_argument(
        "--token_hop_len",
        type=int,
        default=25,
        help="流式处理token跳跃长度：25(默认), 35(平衡), 50(低RTF)。增大可降低RTF但增加首包延迟"
    )
    parser.add_argument(
        "--low_latency",
        action="store_true",
        help="低延迟模式：自动设置flow_timesteps=6，优化RTF（首包延迟不变）"
    )
    parser.add_argument(
        "--low_rtf",
        action="store_true",
        help="低RTF模式：设置flow_timesteps=6, token_hop_len=40，大幅降低RTF（首包延迟略增）"
    )
    
    args = parser.parse_args()
    
    # 低延迟模式覆盖flow_timesteps
    if args.low_latency:
        args.flow_timesteps = 6
        logger.info("启用低延迟模式: flow_timesteps=6")
    
    # 低RTF模式
    if args.low_rtf:
        args.flow_timesteps = 6
        args.token_hop_len = 40
        logger.info("启用低RTF模式: flow_timesteps=6, token_hop_len=40")
    
    # 设置环境变量，供lifespan函数使用
    if args.load_vllm:
        os.environ["LOAD_VLLM"] = "true"
    if args.fp16:
        os.environ["FP16"] = "true"
    os.environ["FLOW_TIMESTEPS"] = str(args.flow_timesteps)
    
    # 如果提供了模型目录，在启动前初始化
    if args.model_dir and os.path.exists(args.model_dir):
        init_model(args.model_dir, load_vllm=args.load_vllm, fp16=args.fp16, 
                   flow_timesteps=args.flow_timesteps, token_hop_len=args.token_hop_len)
    elif args.model_dir:
        # 可能是ModelScope/HuggingFace模型ID，让AutoModel自动下载
        logger.info(f"模型目录不存在，将尝试从ModelScope/HuggingFace下载: {args.model_dir}")
        init_model(args.model_dir, load_vllm=args.load_vllm, fp16=args.fp16, 
                   flow_timesteps=args.flow_timesteps, token_hop_len=args.token_hop_len)
    
    # 配置SSL（如果提供）
    ssl_config = {}
    if args.ssl_keyfile and args.ssl_certfile:
        ssl_config["ssl_keyfile"] = args.ssl_keyfile
        ssl_config["ssl_certfile"] = args.ssl_certfile
        logger.info("启用HTTPS模式")
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        **ssl_config
    )
