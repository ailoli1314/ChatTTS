import sys
import os
import tts_sound_play
from datetime import datetime
import threading
import webrtcvad
import numpy as np
import queue
from tts_sound_play import TTSSoundPlay  # 导入 TTS 播报类

try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_ncnn

def create_recognizer():
    print("Current Working Directory:", os.getcwd())
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
        encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        hotwords_file="",
        hotwords_score=1.5,
    )
    return recognizer


def main():
    print("Started! Please speak")

    # 创建语音识别器
    recognizer = create_recognizer()

    # 设置采样率为 32000 Hz
    sample_rate = 32000

    # 创建 WebRTC VAD 并设置模式
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # 模式0到3，数字越高，语音检测越严格

    # 每次读取 10 ms 的音频样本
    samples_per_read = int(0.01 * sample_rate)  # 320 个样本对应 10ms
    segment_id = 0

    # 使用 `sd.InputStream` 创建一个音频输入流
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            # 从输入流中读取样本，`samples_per_read` 指定读取的样本数
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)

            # 转换样本为 16-bit PCM 格式，因为 WebRTC VAD 要求此格式
            pcm_data = (samples * 32768).astype(np.int16).tobytes()

            # 确保音频帧长度为 10ms，防止帧长度不对引发错误
            if len(pcm_data) == 640:  # 对应 32 kHz 的 20ms
                # 判断当前音频是否包含语音
                is_speech = vad.is_speech(pcm_data, sample_rate)

                if is_speech:
                    recognizer.accept_waveform(sample_rate, samples)
                    is_endpoint = recognizer.is_endpoint
                    result = recognizer.text

                    if is_endpoint and result:
                        print("\r{}|{}:{}".format(datetime.now(), segment_id, result), flush=True)
                        segment_id += 1
                        recognizer.reset()
                        result_array = [result]

                        # 启动任务生成线程
                        producer_thread = threading.Thread(target=task_producer, args=(result_array,))
                        producer_thread.start()

# 创建任务队列
play_queue = queue.Queue()
# 播报线程：从队列中取出音频数据并按顺序播放
def play_worker():
    while True:
        task = play_queue.get()  # 从队列中取出任务
        if task is None:
            break  # 如果收到 None 任务，结束线程
        audio_data, samplerate = task
        sd.play(audio_data, samplerate)  # 播放音频
        sd.wait()  # 等待播放完成
        play_queue.task_done()
# 启动播报线程
play_thread = threading.Thread(target=play_worker)
play_thread.start()
# 创建 TTS 播报对象
tts_sound_play = TTSSoundPlay(play_queue)  # 传入队列
def task_producer(text):
    tts_sound_play.synthesize_and_queue(text)  # 生成音频并放入队列


if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    play_thread.join()