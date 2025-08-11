import pvporcupine
import pvcobra
import numpy as np
import wave
import io
import struct
import logging
from pvrecorder import PvRecorder
from config import *

logger = logging.getLogger("speech_to_speech.voice_recording")

class Recorder:
    def __init__(self, PICOVOICE_KEY):
        self.porcupine = pvporcupine.create(access_key=PICOVOICE_KEY, keyword_paths=["public/Trouble_en_windows_v3_0_0.ppn"])

        self.cobra = pvcobra.create(access_key=PICOVOICE_KEY)

        self.framelength = self.porcupine.frame_length
        self.recorder_device = AUDIO_IN_DEVICE
        self.recorder = PvRecorder(frame_length=self.framelength, device_index=self.recorder_device)

    def record_wake_word(self):
        try:
            self.recorder.start()

            while True:
                frame = self.recorder.read()
                pcm = np.array(frame, dtype=np.int16)
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    logger.debug("Wake word detected")
                    break
        finally:
            self.recorder.stop()
    
    def record_command(self, silence_threshold=SILENCE_THRESHOLD):
        wav_buffer = io.BytesIO()
        wav_file = wave.open(wav_buffer, 'wb')
        wav_file.setparams((1, 2, self.recorder.sample_rate, self.recorder.frame_length, 'NONE', 'NONE'))

        frame_duration = self.framelength / self.recorder.sample_rate
        silence_threshold_sec = silence_threshold
        silence_frames_required = int(silence_threshold_sec / frame_duration)
        silence_frame_count = 0
        total_frame_count = 0

        try:
            self.recorder.start()

            while True:
                frame = self.recorder.read()
                pcm = np.array(frame, dtype=np.int16)
                total_frame_count += pcm.size
                
                wav_file.writeframes(struct.pack("h" * len(frame), *frame))

                voice_prob = self.cobra.process(pcm)

                if voice_prob <= VOICE_PROBABILITY:
                    silence_frame_count += 1
                else:
                    silence_frame_count = 0

                if silence_frame_count >= silence_frames_required:
                    logger.debug("Silence detected, stopping recording")
                    break
        finally:
            self.recorder.stop()
            wav_file.close()

        duration_sec = total_frame_count / self.recorder.sample_rate

        return wav_buffer, duration_sec
    
    def listen_for_wake_word_with_interrupt(self, interrupt_event):
        """Listen for wake word and set interrupt_event when detected"""
        try:
            self.recorder.start()

            while not interrupt_event.is_set():
                frame = self.recorder.read()
                pcm = np.array(frame, dtype=np.int16)
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    logger.debug("Wake word detected during playback")
                    interrupt_event.set()
                    break
        finally:
            self.recorder.stop()