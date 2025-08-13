import io
import time
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from multiprocessing.queues import Queue as QueueClass

from config import DEVICE, WAKEWORD_RESET_TIME, CONTINUATION_THRESHOLD, VOICE_THRESHOLD, SILENCE_THRESHOLD, GOODBYE_PHRASES
from src.logging_config import setup_worker_logging, get_logger
from src.voice_recorder import Recorder
from src.stt_whisper import STTWhisper
from src.utils import save_wav_file
from src.osc import VRChatOSC


class VoiceInputModule:
    def __init__(self, interrupt_count: SynchronizedClass, playback_active: SynchronizedClass, log_queue):
        setup_worker_logging(log_queue)
        self.logger = get_logger("pipeline.voice_input")
        self.interrupt_count = interrupt_count
        self.playback_active = playback_active
        
        self.audio_recorder = Recorder()
        self.whisper = STTWhisper(vad_active=True, device=DEVICE)
        
        self.first_loop = True
        self.ask_wakeword = True
        self.last_command_time = 0
        self.prev_text = ""
        self.osc = VRChatOSC()
    
    def should_reset_wakeword(self) -> bool:
        return (time.time() - self.last_command_time) > WAKEWORD_RESET_TIME
    
    def listen_for_wake_word(self):
        self.logger.debug("Listening for wake word...")
        self.audio_recorder.record_wake_word()
        self.last_command_time = time.time()
        self.ask_wakeword = False
    
    def record_command(self, command_queue: QueueClass) -> tuple[io.BytesIO, str, bool]:
        command_buffer, command_duration = self.audio_recorder.record_command(
            self.ask_wakeword, command_queue, self.interrupt_count
        )
        
        command_buffer.seek(0, io.SEEK_END)
        command_size = command_buffer.tell()
        command_buffer.seek(0)
        
        num_samples = command_size // 2
        if num_samples < self.audio_recorder.porcupine.sample_rate * (VOICE_THRESHOLD + SILENCE_THRESHOLD):
            return None, None, False
        
        return command_buffer, command_duration, True
    
    def transcribe_audio(self, command_buffer: io.BytesIO) -> str:
        output_filename = "command.wav"
        self.logger.debug("Saving wav file.")
        save_wav_file(command_buffer, output_filename, self.logger)
        
        self.logger.debug("Running Speech-To-Text")
        command_buffer.seek(0)
        text_segments = self.whisper.transcribe(command_buffer)
        text = ", ".join([segment.text for segment in text_segments])
        self.logger.info(text)
        
        return text
    
    def is_goodbye_phrase(self, text: str) -> bool:
        """Check if the text contains a goodbye phrase"""
        text_lower = text.lower().strip()
        return any(goodbye.lower() in text_lower for goodbye in GOODBYE_PHRASES)
    
    def process_text_continuation(self, text: str, command_duration: float) -> tuple[str, bool]:
        continuation = False
        
        if (WAKEWORD_RESET_TIME > 0 and not self.first_loop and 
            (time.time() - command_duration - self.last_command_time) < CONTINUATION_THRESHOLD):
            continuation = True
            text = self.prev_text + ", " + text
        
        if not self.ask_wakeword:
            self.last_command_time = time.time()
        
        self.prev_text = text
        self.first_loop = False
        
        return text, continuation
    
    def run_loop(self, voice_setup_event: EventClass, pipeline_setup_event: EventClass, 
                 command_queue: QueueClass):
        voice_setup_event.set()
        self.logger.debug("Waiting for Pipeline to setup")
        pipeline_setup_event.wait()
        
        while True:
                
            if self.should_reset_wakeword():
                self.logger.warning("wakeword reset")
                self.ask_wakeword = True
                self.osc.clear_message()
            
            if self.ask_wakeword:
                self.listen_for_wake_word()
            
            if self.ask_wakeword and not command_queue.empty():
                self.logger.warning("interrupt fired")
                self.interrupt_count.value += 1
            
            self.logger.debug("Listening for command...")
            command_buffer, command_duration, has_speech = self.record_command(command_queue)
            
            if not has_speech:
                self.logger.debug("No speech detected.")
                continue
            
            text = self.transcribe_audio(command_buffer)
            
            if not text:
                self.logger.debug("No command detected")
                continue
            
            text, continuation = self.process_text_continuation(text, command_duration)
            
            # Check if this is a goodbye phrase and reset to wake word mode
            is_goodbye = self.is_goodbye_phrase(text)
            
            # Add timing information for benchmarking
            command_start_time = time.time()
            
            command_queue.put({"text": text, "marker": "start", "continuation": continuation, "is_goodbye": is_goodbye, "start_time": command_start_time})
            command_queue.put({"text": text, "marker": "finish", "continuation": continuation, "is_goodbye": is_goodbye, "start_time": command_start_time})
            
            # If goodbye phrase detected, reset conversation state
            if is_goodbye:
                self.logger.info("Goodbye detected - resetting to wake word mode")
                self.ask_wakeword = True
                self.first_loop = True
                self.prev_text = ""
                self.last_command_time = 0