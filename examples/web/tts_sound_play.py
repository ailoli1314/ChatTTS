import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import ChatTTS
import torch
import torchaudio
from tools.audio import float_to_int16
import numpy as np
import sounddevice as sd


chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["早上好，主人"]

# wavs = chat.infer(texts)
#
# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7)
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

# 假设 `float_to_int16(wavs[0]).T` 生成的是一维音频数据
audio_data = float_to_int16(wavs[0]).T

# 播放音频，注意数据类型需要是 int16
sd.play(audio_data, samplerate=24000)

# 等待音频播放完成
sd.wait()

# # 如果 audio_data 是一维的，需要将其转换为二维 (1, num_samples)
# if audio_data.ndim == 1:
#     audio_data = np.expand_dims(audio_data, 0)  # 添加一个通道维度
#
# # 将 numpy 数组转换为 PyTorch 张量
# audio_tensor = torch.from_numpy(audio_data)
# # 将音频保存为 wav 文件
# samplerate = 24000
# torchaudio.save("output.wav", torch.from_numpy(audio_data), samplerate)