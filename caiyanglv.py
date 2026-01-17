import librosa
import soundfile as sf

# 配置文件路径
input_wav_path = r"D:\CosyVoice-finally\asset\zh_male.wav"  # 原8000Hz文件
output_wav_path = r"D:\CosyVoice-finally\asset\zh_male1_24000.wav"  # 转换后24000Hz文件
target_sr = 24000  # 目标采样率（符合CosyVoice要求）

# 加载音频文件（自动读取原采样率）
y, sr = librosa.load(input_wav_path, sr=None)  # sr=None保留原采样率读取

# 重采样到24000Hz
y_24000 = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

# 保存转换后的音频（保持WAV格式，无损编码）
sf.write(output_wav_path, y_24000, target_sr)
print(f"转换完成！新文件采样率：{target_sr}Hz，保存路径：{output_wav_path}")