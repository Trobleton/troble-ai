from multiprocessing import Process, Queue, Event, Value
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from multiprocessing.queues import Queue as QueueClass

from src.logging_config import setup_worker_logging, get_logger
from src.pipeline.voice_input_module import VoiceInputModule
from src.pipeline.intelligence_module import IntelligenceModule
from src.pipeline.audio_output_module import AudioOutputModule


def voice_worker_func(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, log_queue):
    voice_module = VoiceInputModule(interrupt_count, log_queue)
    voice_module.run_loop(voice_setup_event, pipeline_setup_event, command_queue)


def pipeline_worker_func(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, log_queue):
    setup_worker_logging(log_queue)
    logger = get_logger("pipeline.worker")
    
    intelligence_module = IntelligenceModule(interrupt_count, log_queue)
    audio_output_module = AudioOutputModule(interrupt_count, log_queue)
    
    pipeline_setup_event.set()
    logger.debug("Waiting for voice recording to setup")
    voice_setup_event.wait()
    
    while True:
        if not command_queue.empty():
            work = command_queue.get()
            logger.debug(work)
            
            text = work["text"]
            marker = work["marker"]
            continuation = work["continuation"]
            is_goodbye = work.get("is_goodbye", False)
            
            if marker == "start":
                response, interrupted = intelligence_module.process_query(text, continuation, is_goodbye)
                
                if not interrupted and response:
                    interrupted = audio_output_module.synthesize_and_play(
                        response, save_file="output.wav"
                    )
                
                if not interrupted:
                    intelligence_module.llm.interrupt_context.clear()
                
                # Always consume the finish marker to maintain queue sync
                finish_work = command_queue.get()  # Remove finish marker
                
                # If processing was interrupted, still need to decrement interrupt count
                if interrupted and finish_work["marker"] == "finish":
                    interrupt_count.value -= 1
                    
            elif marker == "finish":
                # This should only happen if we somehow get a finish marker without start
                if interrupt_count.value > 0:
                    interrupt_count.value -= 1


class PipelineOrchestrator:
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.logger = get_logger("pipeline.orchestrator")
        
        self.command_queue = Queue()
        self.interrupt_event = Event()
        self.interrupt_count = Value("i", 0)
        self.voice_setup_event = Event()
        self.pipeline_setup_event = Event()
        
        self.voice_worker_process = None
        self.pipeline_worker_process = None
    
    def start(self):
        self.logger.debug("Starting Processes")
        
        self.voice_worker_process = Process(
            target=voice_worker_func, 
            args=(self.interrupt_count, self.voice_setup_event, self.pipeline_setup_event, self.command_queue, self.log_queue)
        )
        self.pipeline_worker_process = Process(
            target=pipeline_worker_func, 
            args=(self.interrupt_count, self.voice_setup_event, self.pipeline_setup_event, self.command_queue, self.log_queue)
        )
        
        self.voice_worker_process.start()
        self.pipeline_worker_process.start()
    
    def wait_for_completion(self):
        if self.voice_worker_process:
            self.voice_worker_process.join()
        if self.pipeline_worker_process:
            self.pipeline_worker_process.join()
        self.logger.debug("Processes Completed")
    
    def stop(self):
        if self.voice_worker_process and self.voice_worker_process.is_alive():
            self.voice_worker_process.terminate()
        if self.pipeline_worker_process and self.pipeline_worker_process.is_alive():
            self.pipeline_worker_process.terminate()