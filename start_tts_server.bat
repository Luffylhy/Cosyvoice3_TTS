@echo off
REM TTS API服务器启动脚本 (Windows)

REM 设置默认值
if "%MODEL_DIR%"=="" set MODEL_DIR=models\Fun-CosyVoice3-0.5B-2512
if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=0.0.0.0

REM 检查模型目录是否存在
if not exist "%MODEL_DIR%" (
    echo 警告: 模型目录不存在: %MODEL_DIR%
    echo 请设置 MODEL_DIR 环境变量或使用 --model_dir 参数
    echo 例如: set MODEL_DIR=models\Fun-CosyVoice3-0.5B-2512
)

REM 启动服务器
echo 正在启动TTS API服务器...
echo 模型目录: %MODEL_DIR%
echo 监听地址: %HOST%:%PORT%

python tts_api_server.py --model_dir "%MODEL_DIR%" --host %HOST% --port %PORT% %*
