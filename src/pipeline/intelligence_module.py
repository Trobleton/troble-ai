import threading
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

from config import TTS_CHOICE, RAG_CONFIDENCE_THRESHOLD, ENABLE_WEBSEARCH
from src.logging_config import setup_worker_logging, get_logger
from src.llm_wrapper import LLMWrapper
from src.rag_langchain import RAGLangchain
from src.web_search import WebSearcher
from src.tts_orpheus import TTSOrpheus
from src.tts_kokoro import TTSKokoro
from src.utils import play_wav_file


class IntelligenceModule:
    def __init__(self, interrupt_count: SynchronizedClass, log_queue):
        setup_worker_logging(log_queue)
        self.logger = get_logger("pipeline.intelligence")
        self.interrupt_count = interrupt_count
        
        self.llm = LLMWrapper(interrupt_count=interrupt_count)
        self.websearch = WebSearcher(interrupt_count=interrupt_count)
        self.rag = RAGLangchain(interrupt_count=interrupt_count)
        
        if TTS_CHOICE == "orpheus":
            self.tts = TTSOrpheus(interrupt_count=interrupt_count)
        elif TTS_CHOICE == "kokoro":
            self.tts = TTSKokoro(interrupt_count=interrupt_count)
    
    def interrupt_actions(self, text: str, info: str):
        self.logger.warning(f"Pipeline Interrupted: {info}")
        self.llm.interrupt_context.append(text)
    
    def play_search_notification(self, topic: str):
        def searching_speech_worker():
            output_buffer, output_duration = self.tts.synthesize(f"Searching the web for {topic}")
            output_buffer.seek(0)
            play_wav_file(output_buffer, self.logger, self.interrupt_count)
        
        speech_thread = threading.Thread(target=searching_speech_worker)
        speech_thread.start()
        return speech_thread
    
    def query_rag(self, text: str) -> tuple[list, bool]:
        self.logger.debug("Querying RAG")
        query_results = self.rag.query(text)
        
        if self.interrupt_count.value > 0:
            return [], True
        
        query_result_scores = [query["score"] for query in query_results] + [0]
        return query_results, max(query_result_scores) >= RAG_CONFIDENCE_THRESHOLD
    
    def perform_websearch(self, text: str, topic: str) -> tuple[list, bool]:
        speech_thread = self.play_search_notification(topic)
        
        websites = self.websearch.ddg_search(text)
        if self.interrupt_count.value > 0:
            return [], True
        
        self.logger.debug("Fetching website contents")
        web_contents = self.websearch.fetch_content(websites)
        if self.interrupt_count.value > 0:
            return [], True
        
        self.logger.debug("Adding contents to RAG")
        for document in web_contents:
            if self.interrupt_count.value > 0:
                return [], True
            self.rag.add_document(document)
        
        self.logger.debug("Querying RAG Again")
        query_results = self.rag.query(topic)
        self.logger.info(query_results)
        
        if self.interrupt_count.value > 0:
            return [], True
        
        speech_thread.join()
        return query_results, False
    
    def decide_websearch_needed(self, text: str) -> tuple[bool, str, bool]:
        self.logger.debug("Not enough confident info in RAG, decide search")
        decision, topic = self.llm.decide_websearch(text)
        
        if self.interrupt_count.value > 0:
            return False, "", True
        
        self.logger.debug(f"Websearch recommended?: {decision} - {topic}")
        return decision == "yes", topic, False
    
    def generate_response(self, text: str, context: str) -> tuple[str, bool]:
        self.logger.debug("Sending to LLM")
        response = self.llm.send_to_llm(text, context)
        
        if self.interrupt_count.value > 0:
            return "", True
        
        self.logger.info(response)
        return response, False
    
    def process_query(self, text: str, continuation: bool, is_goodbye: bool = False) -> tuple[str, bool]:
        if continuation:
            self.llm.interrupt_context.pop()
        
        if self.interrupt_count.value > 0:
            self.interrupt_actions(text, info="before pipeline start")
            return "", True
        
        # Handle goodbye phrases with a simple response
        if is_goodbye:
            goodbye_responses = [
                "Goodbye! Say 'Trouble' to talk again.",
                "See you later! Wake me with 'Trouble' when you need me.",
                "Take care! Just say 'Trouble' to start another conversation.",
                "Bye! I'll be listening for 'Trouble' when you're ready to chat.",
                "Until next time! Remember to say 'Trouble' to wake me up."
            ]
            import random
            response = random.choice(goodbye_responses)
            return response, False
        
        query_results, has_confident_info = self.query_rag(text)
        if self.interrupt_count.value > 0:
            self.interrupt_actions(text, info="querying RAG")
            return "", True
        
        if not has_confident_info and ENABLE_WEBSEARCH:
            needs_search, topic, interrupted = self.decide_websearch_needed(text)
            if interrupted:
                self.interrupt_actions(text, info="Decide websearch")
                return "", True
            
            if needs_search:
                query_results, interrupted = self.perform_websearch(text, topic)
                if interrupted:
                    self.interrupt_actions(text, info="Performing websearch")
                    return "", True
        
        self.logger.debug("Info in RAG exists, no search needed")
        
        context = ""
        for result in query_results:
            context += result["content"]
        
        response, interrupted = self.generate_response(text, context)
        if interrupted:
            self.interrupt_actions(text, info="LLM Query")
            return "", True
        
        return response, False