#!/bin/bash
# TTS API服务器启动脚本

# 设置默认值
MODEL_DIR="${MODEL_DIR:-models/Fun-CosyVoice3-0.5B-2512}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "警告: 模型目录不存在: $MODEL_DIR"
    echo "请设置 MODEL_DIR 环境变量或使用 --model_dir 参数"
    echo "例如: export MODEL_DIR=models/Fun-CosyVoice3-0.5B-2512"
fi

# 启动服务器
echo "正在启动TTS API服务器..."
echo "模型目录: $MODEL_DIR"
echo "监听地址: $HOST:$PORT"

python tts_api_server.py \
    --model_dir "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
