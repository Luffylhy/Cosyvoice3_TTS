#!/bin/bash
# 快速流式测试脚本 (Linux/Mac)

echo "========================================"
echo "实时流式TTS测试"
echo "========================================"
echo

# 设置服务器地址（请修改为您的服务器地址）
SERVER_URL="${SERVER_URL:-http://localhost:8000}"

echo "测试服务器: $SERVER_URL"
echo

# 基本测试
echo "[1/3] 基本流式测试..."
python test_streaming_tts.py --server "$SERVER_URL" --text "Hello, this is a streaming test." --output test1.wav
echo

# 实时播放测试（需要pyaudio）
echo "[2/3] 实时播放测试..."
python test_streaming_tts.py --server "$SERVER_URL" --text "Testing real-time playback." --play --output test2.wav
echo

# 性能统计测试
echo "[3/3] 性能统计测试..."
python test_streaming_tts.py --server "$SERVER_URL" --text "This is a longer text to test the streaming performance and latency of the TTS system. We want to see how well it handles continuous audio streaming." --output test3.wav
echo

echo "========================================"
echo "测试完成！"
echo "========================================"
