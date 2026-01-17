# -*- coding: utf-8 -*-
"""
TTS API客户端示例
演示如何调用TTS API服务器进行语音合成
"""
import requests
import argparse
import os
import wave
import numpy as np


def save_audio_stream(response, output_file, sample_rate=22050):
    """
    保存音频流到文件
    
    Args:
        response: requests响应对象（stream=True）
        output_file: 输出文件路径
        sample_rate: 采样率（从响应头获取或使用默认值）
    """
    # 从响应头获取采样率
    if 'X-Sample-Rate' in response.headers:
        sample_rate = int(response.headers['X-Sample-Rate'])
    
    # 收集所有音频数据
    audio_data = b''
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            audio_data += chunk
    
    # 转换为numpy数组
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # 保存为WAV文件
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位 = 2字节
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    print(f"音频已保存到: {output_file}")
    print(f"采样率: {sample_rate} Hz")
    print(f"时长: {len(audio_array) / sample_rate:.2f} 秒")


def call_tts_api(
    server_url: str,
    text: str,
    language: str = "en",
    emotion: str = "professional",
    gender: str = "male",
    output_file: str = "output.wav"
):
    """
    调用TTS API
    
    Args:
        server_url: 服务器URL（例如: http://localhost:8000）
        text: 要合成的文本
        language: 语言代码 (en/ru/fr)
        emotion: 情绪 (professional/friendly/excited)
        gender: 性别 (male/female)
        output_file: 输出文件路径
    """
    url = f"{server_url}/tts"
    
    data = {
        "text": text,
        "language": language,
        "emotion": emotion,
        "gender": gender,
        "stream": True
    }
    
    print(f"正在请求TTS服务...")
    print(f"URL: {url}")
    print(f"文本: {text}")
    print(f"语言: {language}, 情绪: {emotion}, 性别: {gender}")
    
    try:
        response = requests.post(url, data=data, stream=True, timeout=300)
        response.raise_for_status()
        
        # 保存音频
        save_audio_stream(response, output_file)
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")
        raise


def main():
    parser = argparse.ArgumentParser(description="TTS API客户端示例")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="TTS服务器地址"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="要合成的文本"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ru", "fr"],
        help="语言代码"
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="professional",
        choices=["professional", "friendly", "excited"],
        help="情绪"
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=["male", "female"],
        help="性别"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="输出音频文件路径"
    )
    
    args = parser.parse_args()
    
    call_tts_api(
        args.server,
        args.text,
        args.language,
        args.emotion,
        args.gender,
        args.output
    )


if __name__ == "__main__":
    main()
