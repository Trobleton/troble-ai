import wave
import logging
import io
import os

# Add CUDA DLL directory before importing PyTorch
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

import numpy as np
import torch
from config import *
from kokoro import KPipeline

logger = logging.getLogger("speech_to_speech.tts")

class TTSKokoro:
    def __init__(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

        self.client = KPipeline(lang_code=KOKORO_TTS_LANG, device=DEVICE, repo_id="hexgrad/Kokoro-82M")
        self.voice = KOKORO_TTS_VOICE
        self.samplerate = 24000

    def synthesize(self, text):
        # Prepend a space to help TTS models not cut off the first character
        text = " " + text
        wav_buffer = io.BytesIO()
        wav_file = wave.open(wav_buffer, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.samplerate)

        audio_duration = 0

        try:
            # Add 200ms of silence at the start to prevent cutoff
            silence = np.zeros(int(0.2 * self.samplerate), dtype=np.int16)
            wav_file.writeframes(silence.tobytes())

            generator = self.client(text, voice=self.voice)

            for _, _, audio in generator:
                audio = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                audio = audio.numpy()
                audio *= 32767

                audio_duration = len(audio) / 24000
                wav_data = audio.astype(np.int16, copy=False)
                wav_file.writeframes(wav_data.tobytes())
        finally:
            wav_file.close()

        return wav_buffer, audio_duration