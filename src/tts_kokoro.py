import wave
import logging
import io
import os
import numpy as np
import warnings

# Suppress common PyTorch warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last")
warnings.filterwarnings("ignore", message="torch.nn.utils.weight_norm is deprecated")

import torch
from config import *
from kokoro import KPipeline
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

# Add CUDA DLL directory before importing PyTorch
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

class TTSKokoro:
  def __init__(self, interrupt_count: SynchronizedClass):
    torch.backends.cudnn.benchmark = False  # prevents unsupported plan attempts
    torch.backends.cuda.matmul.allow_tf32 = True  # small perf boost on Ampere+
    
    self.logger = logging.getLogger("speech_to_speech.tts_kokoro")
    self.interrupt_count = interrupt_count
    
    self.client = KPipeline(lang_code=KOKORO_TTS_LANG, device=DEVICE, repo_id='hexgrad/Kokoro-82M') # lang code a = american, b = british
    self.voice = KOKORO_TTS_VOICE
    self.samplerate = 24000
 
  def synthesize(self, text):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(self.samplerate)
    
    audio_duration = 0
    
    try:
      generator = self.client(text, voice=self.voice, speed=1.3 if self.voice == "af_nicole" else 1.0)
      
      chunk_count = 0
      for _, _,audio in generator:
        chunk_count += 1
        if chunk_count <= 3:  # Log first few chunks
          self.logger.debug(f"Kokoro chunk {chunk_count}: audio shape={audio.shape if hasattr(audio, 'shape') else len(audio) if audio is not None else 'None'}")
          if chunk_count == 1:
            self.logger.debug(f"First chunk content preview: {audio[:10] if hasattr(audio, '__len__') and len(audio) > 10 else audio}")
        if self.interrupt_count.value > 0:
          break

        audio = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
        audio = audio.numpy()
        audio *= 32767
        
        audio_duration = len(audio) / self.samplerate
        wav_Data = audio.astype(np.int16, copy=False)
        wav_file.writeframes(wav_Data.tobytes())

    finally:
      wav_file.close()
    
    return wav_buffer, audio_duration