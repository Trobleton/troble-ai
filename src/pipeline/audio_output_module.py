import io
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

from config import TTS_CHOICE
from src.logging_config import setup_worker_logging, get_logger
from src.tts.kokoro import TTSKokoro
from src.utils import save_wav_file, play_wav_file
from src.osc import VRChatOSC


class AudioOutputModule:
    def __init__(self, interrupt_count: SynchronizedClass, playback_active: SynchronizedClass, log_queue):
        setup_worker_logging(log_queue)
        self.logger = get_logger("pipeline.audio_output")
        self.interrupt_count = interrupt_count
        self.playback_active = playback_active
        self.osc = VRChatOSC()
        self.tts = TTSKokoro(interrupt_count=interrupt_count)
    
    def synthesize_and_play(self, text: str, save_file: str = None) -> bool:
        self.logger.debug("Synthesizing speech")
        output_buffer, output_duration = self.tts.synthesize(text)
        
        if self.interrupt_count.value > 0:
            return True
        
        output_buffer.seek(0)
        
        if save_file:
            self.logger.debug("Saving wav file.")
            save_wav_file(output_buffer, save_file, self.logger)
            output_buffer.seek(0)
        
        self.logger.debug("Playing response")
        self.osc.send_message(text)
        play_wav_file(output_buffer, self.logger, interrupt_count=self.interrupt_count)
        
        if self.interrupt_count.value > 0:
            return True
        
        return False
    
    def synthesize_only(self, text: str) -> tuple[io.BytesIO, float, bool]:
        self.logger.debug("Synthesizing speech")
        output_buffer, output_duration = self.tts.synthesize(text)
        
        if self.interrupt_count.value > 0:
            return None, 0, True
        
        return output_buffer, output_duration, False