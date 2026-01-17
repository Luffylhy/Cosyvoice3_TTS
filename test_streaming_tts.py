# -*- coding: utf-8 -*-
"""
实时流式TTS测试脚本
用于测试远程服务器的流式音频输出效果
支持实时播放、保存和分析流式传输性能
"""
import requests
import argparse
import time
import wave
import numpy as np
import sys
from io import BytesIO


def get_voice_name(gender: str, voice_id: str) -> str:
    """
    获取音色名称和描述
    
    Args:
        gender: 性别 (male/female)
        voice_id: 音色ID (1/2/3)
    
    Returns:
        音色名称和描述字符串
    """
    voice_names = {
        "male": {
            "1": "孙悟空(西北方言)",
            "2": "猪八戒",
            "3": "太乙真人(四川方言)"
        },
        "female": {
            "1": "武则天",
            "2": "林志玲(台湾口音)",
            "3": "东北雨姐(东北口音)"
        }
    }
    gender_key = "male" if gender.lower() in ["male", "男"] else "female"
    return voice_names.get(gender_key, {}).get(str(voice_id), f"音色{voice_id}")


def test_streaming_tts(
    server_url: str,
    text: str,
    language: str = "en",
    emotion: str = "professional",
    gender: str = "male",
    voice_id: str = "1",
    output_file: str = None,
    play_audio: bool = False,
    show_stats: bool = True
):
    """
    测试流式TTS API
    
    Args:
        server_url: 服务器URL
        text: 要合成的文本
        language: 语言代码
        emotion: 情绪
        gender: 性别
        voice_id: 音色ID (1/2/3)
            - 男声: 1=孙悟空(西北方言), 2=猪八戒, 3=太乙真人(四川方言)
            - 女声: 1=武则天, 2=林志玲(台湾口音), 3=东北雨姐(东北口音)
        output_file: 输出文件路径（可选）
        play_audio: 是否实时播放音频
        show_stats: 是否显示统计信息
    """
    url = f"{server_url}/tts"
    
    data = {
        "text": text,
        "language": language,
        "emotion": emotion,
        "gender": gender,
        "voice_id": voice_id,
        "stream": True
    }
    
    # 获取音色名称
    voice_name = get_voice_name(gender, voice_id)
    
    print("=" * 60)
    print("实时流式TTS测试")
    print("=" * 60)
    print(f"服务器: {server_url}")
    print(f"文本: {text}")
    print(f"语言: {language}, 情绪: {emotion}, 性别: {gender}, 音色: {voice_id} ({voice_name})")
    print("-" * 60)
    
    # 统计信息
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    audio_chunks = []
    
    try:
        print("正在发送请求...")
        response = requests.post(url, data=data, stream=True, timeout=300)
        response.raise_for_status()
        
        # 获取采样率
        sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
        print(f"采样率: {sample_rate} Hz")
        print("-" * 60)
        print("开始接收音频流...")
        print()
        
        # 实时接收音频流
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                chunk_time = time.time()
                
                # 记录第一个数据块的时间
                if first_chunk_time is None:
                    first_chunk_time = chunk_time
                    first_chunk_latency = first_chunk_time - start_time
                    print(f"✓ 首包延迟: {first_chunk_latency:.3f} 秒")
                    print()
                
                total_bytes += len(chunk)
                chunk_count += 1
                audio_chunks.append(chunk)
                
                # 实时播放（如果启用）
                if play_audio:
                    try:
                        import pyaudio
                        if chunk_count == 1:
                            # 初始化PyAudio（只在第一次）
                            p = pyaudio.PyAudio()
                            stream = p.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=sample_rate,
                                output=True
                            )
                            print("开始实时播放音频...")
                        
                        stream.write(chunk)
                    except ImportError:
                        print("警告: pyaudio未安装，无法实时播放。使用 'pip install pyaudio' 安装")
                        play_audio = False
                    except Exception as e:
                        print(f"播放错误: {e}")
                        play_audio = False
                
                # 显示进度
                if show_stats and chunk_count % 10 == 0:
                    elapsed = chunk_time - start_time
                    if elapsed > 0:
                        speed = total_bytes / elapsed / 1024  # KB/s
                        print(f"已接收: {chunk_count} 块, {total_bytes/1024:.1f} KB, "
                              f"速度: {speed:.1f} KB/s", end='\r')
        
        # 关闭音频流
        if play_audio and 'stream' in locals():
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("\n播放完成")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 合并所有音频块
        audio_data = b''.join(audio_chunks)
        
        print()
        print("-" * 60)
        print("流式传输完成")
        print("-" * 60)
        
        # 显示统计信息
        if show_stats:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_duration = len(audio_array) / sample_rate
            
            print(f"总传输时间: {total_duration:.3f} 秒")
            if first_chunk_time:
                print(f"首包延迟: {first_chunk_latency:.3f} 秒")
            print(f"音频时长: {audio_duration:.2f} 秒")
            print(f"传输速度: {total_bytes / total_duration / 1024:.1f} KB/s")
            print(f"数据块数量: {chunk_count}")
            print(f"总数据量: {total_bytes / 1024:.1f} KB")
            if total_duration > 0:
                print(f"实时因子 (RTF): {total_duration / audio_duration:.2f}x")
                if first_chunk_time:
                    print(f"首包后延迟: {total_duration - first_chunk_latency:.3f} 秒")
        
        # 保存音频文件
        if output_file:
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            print(f"\n音频已保存到: {output_file}")
        
        print("=" * 60)
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"错误详情: {e.response.text}")
        return False
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return False
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_requests(server_url: str, count: int = 5):
    """测试多个并发请求"""
    print("=" * 60)
    print(f"并发测试 - {count} 个请求")
    print("=" * 60)
    
    import concurrent.futures
    
    def make_request(i):
        text = f"This is test request number {i+1}."
        return test_streaming_tts(
            server_url, text, "en", "professional", "male",
            output_file=f"test_output_{i+1}.wav",
            play_audio=False,
            show_stats=False
        )
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(make_request, i) for i in range(count)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    success_count = sum(results)
    
    print(f"\n并发测试完成:")
    print(f"成功: {success_count}/{count}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均耗时: {(end_time - start_time) / count:.2f} 秒/请求")


def main():
    parser = argparse.ArgumentParser(description="实时流式TTS测试工具")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="TTS服务器地址（例如: http://your-server.com:8000）"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the streaming text-to-speech system. The system supports real-time audio streaming with low latency.",
        help="要合成的文本"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ru", "fr", "zh"],
        help="语言代码: en(英语), ru(俄语), fr(法语), zh(中文)"
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="professional",
        choices=["professional", "friendly", "excited", "专业", "亲切", "兴奋"],
        help="情绪: professional(专业), friendly(亲切), excited(兴奋)"
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=["male", "female", "男", "女"],
        help="性别: male(男声), female(女声)"
    )
    parser.add_argument(
        "--voice_id",
        type=str,
        default="1",
        choices=["1", "2", "3"],
        help="音色ID: 1, 2, 3（默认: 1）。男声: 1=孙悟空(西北方言), 2=猪八戒, 3=太乙真人(四川方言)。女声: 1=武则天, 2=林志玲(台湾口音), 3=东北雨姐(东北口音)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="streaming_test.wav",
        help="输出音频文件路径"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="实时播放音频（需要pyaudio）"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="不显示统计信息"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=0,
        help="并发测试请求数量（0表示不进行并发测试）"
    )
    
    args = parser.parse_args()
    
    # 并发测试
    if args.concurrent > 0:
        test_multiple_requests(args.server, args.concurrent)
    else:
        # 单次测试
        test_streaming_tts(
            args.server,
            args.text,
            args.language,
            args.emotion,
            args.gender,
            args.voice_id,
            args.output,
            args.play,
            not args.no_stats
        )


if __name__ == "__main__":
    main()
