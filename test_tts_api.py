# -*- coding: utf-8 -*-
"""
TTS API测试脚本
用于快速测试TTS API的各项功能
"""
import requests
import wave
import numpy as np
import os
import sys


def test_health_check(server_url):
    """测试健康检查端点"""
    print("=" * 50)
    print("测试健康检查端点...")
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 健康检查通过: {result}")
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def test_tts_basic(server_url, output_dir="test_outputs"):
    """测试基本TTS功能"""
    print("=" * 50)
    print("测试基本TTS功能...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_cases = [
        {
            "name": "英语-专业-男声",
            "text": "Hello, this is a test of the text-to-speech system. The system supports multiple languages, emotions, and voice genders.",
            "language": "en",
            "emotion": "professional",
            "gender": "male",
            "output": f"{output_dir}/en_professional_male.wav"
        },
        {
            "name": "英语-亲切-女声",
            "text": "Welcome! I'm here to help you with a warm and friendly voice.",
            "language": "en",
            "emotion": "friendly",
            "gender": "female",
            "output": f"{output_dir}/en_friendly_female.wav"
        },
        {
            "name": "英语-兴奋-男声",
            "text": "This is amazing! I'm so excited to demonstrate this incredible technology!",
            "language": "en",
            "emotion": "excited",
            "gender": "male",
            "output": f"{output_dir}/en_excited_male.wav"
        },
        {
            "name": "俄语-专业-女声",
            "text": "Добро пожаловать в наш сервис синтеза речи. Мы предоставляем высококачественный синтез речи.",
            "language": "ru",
            "emotion": "professional",
            "gender": "female",
            "output": f"{output_dir}/ru_professional_female.wav"
        },
        {
            "name": "法语-亲切-男声",
            "text": "Bienvenue dans notre service de synthèse vocale. Nous sommes là pour vous aider.",
            "language": "fr",
            "emotion": "friendly",
            "gender": "male",
            "output": f"{output_dir}/fr_friendly_male.wav"
        },
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 测试: {test_case['name']}")
        try:
            url = f"{server_url}/tts"
            data = {
                "text": test_case["text"],
                "language": test_case["language"],
                "emotion": test_case["emotion"],
                "gender": test_case["gender"],
                "stream": True
            }
            
            response = requests.post(url, data=data, stream=True, timeout=300)
            response.raise_for_status()
            
            # 保存音频
            audio_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    audio_data += chunk
            
            if len(audio_data) > 0:
                # 获取采样率
                sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
                
                # 保存为WAV文件
                with wave.open(test_case["output"], 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                
                # 计算时长
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                duration = len(audio_array) / sample_rate
                
                print(f"  ✓ 成功 - 保存到: {test_case['output']}")
                print(f"    时长: {duration:.2f}秒, 采样率: {sample_rate}Hz")
                success_count += 1
            else:
                print(f"  ✗ 失败 - 未收到音频数据")
                
        except Exception as e:
            print(f"  ✗ 失败 - {e}")
    
    print(f"\n测试完成: {success_count}/{len(test_cases)} 通过")
    return success_count == len(test_cases)


def test_error_handling(server_url):
    """测试错误处理"""
    print("=" * 50)
    print("测试错误处理...")
    
    error_cases = [
        {
            "name": "无效语言",
            "data": {
                "text": "Test",
                "language": "invalid_lang",
                "emotion": "professional",
                "gender": "male"
            },
            "expected_status": 400
        },
        {
            "name": "无效情绪",
            "data": {
                "text": "Test",
                "language": "en",
                "emotion": "invalid_emotion",
                "gender": "male"
            },
            "expected_status": 400
        },
        {
            "name": "缺少文本",
            "data": {
                "language": "en",
                "emotion": "professional",
                "gender": "male"
            },
            "expected_status": 422  # FastAPI validation error
        },
    ]
    
    success_count = 0
    for i, test_case in enumerate(error_cases, 1):
        print(f"\n[{i}/{len(error_cases)}] 测试: {test_case['name']}")
        try:
            response = requests.post(f"{server_url}/tts", data=test_case["data"], timeout=10)
            if response.status_code == test_case["expected_status"]:
                print(f"  ✓ 成功 - 返回预期错误码: {response.status_code}")
                success_count += 1
            else:
                print(f"  ✗ 失败 - 预期错误码 {test_case['expected_status']}, 实际: {response.status_code}")
        except Exception as e:
            print(f"  ✗ 失败 - {e}")
    
    print(f"\n错误处理测试: {success_count}/{len(error_cases)} 通过")
    return success_count == len(error_cases)


def main():
    """主测试函数"""
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("TTS API 测试套件")
    print(f"服务器地址: {server_url}")
    print("=" * 50)
    
    # 测试健康检查
    if not test_health_check(server_url):
        print("\n服务器未就绪，请先启动TTS服务器")
        return
    
    # 测试基本TTS功能
    test_tts_basic(server_url)
    
    # 测试错误处理
    test_error_handling(server_url)
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print(f"测试输出文件保存在: test_outputs/")


if __name__ == "__main__":
    main()
