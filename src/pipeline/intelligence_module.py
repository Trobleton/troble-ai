import time
import re
import random
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

from config import RAG_CONFIDENCE_THRESHOLD, ENABLE_WEBSEARCH
from src.logging_config import setup_worker_logging, get_logger
from src.llm_wrapper import LLMWrapper
from src.rag_langchain import RAGLangchain
from src.web_search import WebSearcher
from src.tts.kokoro import TTSKokoro
from src.utils import play_wav_file
from src.osc import VRChatOSC

class IntelligenceModule:
    def __init__(self, interrupt_count: SynchronizedClass, playback_active: SynchronizedClass, log_queue):
        setup_worker_logging(log_queue)
        self.logger = get_logger("pipeline.intelligence")
        self.interrupt_count = interrupt_count
        self.playback_active = playback_active
        self.osc = VRChatOSC()
        
        self.llm = LLMWrapper(interrupt_count=interrupt_count)
        self.websearch = WebSearcher(interrupt_count=interrupt_count)
        self.rag = RAGLangchain(interrupt_count=interrupt_count)
        
        # Track last processed query to prevent loops
        self.last_processed_query = ""
        self.last_processed_time = 0
        
        self.tts = TTSKokoro(interrupt_count=interrupt_count)
    
    def interrupt_actions(self, text: str, info: str):
        self.logger.warning(f"Pipeline Interrupted: {info}")
        # Only add to interrupt context if not already there to prevent duplicates
        if text not in self.llm.interrupt_context:
            self.llm.interrupt_context.append(text)
    
    def play_search_notification(self, topic: str):
        self.playback_active.value = 1  # Disable listening during search notification
        
        output_buffer, output_duration = self.tts.synthesize(f"Searching the web for {topic}")
        if self.interrupt_count.value > 0:
            self.playback_active.value = 0
            return
            
        output_buffer.seek(0)
        play_wav_file(output_buffer, self.logger, self.interrupt_count)
        
        self.playback_active.value = 0  # Re-enable listening after notification
    
    def _is_meaningful_prompt(self, text: str) -> bool:
        """Check if a prompt is meaningful enough to warrant a response."""
        if not text:
            return False
        
        # Clean the text for analysis
        cleaned_text = text.lower().strip()
        words = cleaned_text.split()
        
        # Always respond if longer than 2 words
        if len(words) > 2:
            return True
        
        # Filter out common meaningless short phrases
        meaningless_patterns = [
            # Single characters or sounds
            r'^[a-z]$',  # Single letters
            r'^(uh|um|hmm|ah|oh|eh)$',  # Filler sounds
            r'^(huh|what|hey|yo)$',  # Short exclamations that need context
            
            # Very short incomplete phrases
            r'^(the|and|or|but|if|so|to|a|an|is|are|was|were|be)$',  # Articles/conjunctions alone
            r'^(yes|no|ok|okay)$',  # Simple affirmations without context
            r'^(hi|hello)$',  # Greetings without questions
            
            # Common transcription errors
            r'^\.$',  # Just punctuation
            r'^,$',
            r'^\?$',
            r'^!$',
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, cleaned_text):
                return False
        
        # Check for meaningful 2-word phrases
        if len(words) == 2:
            meaningful_2_word_patterns = [
                r'(what|how|when|where|why|who|can|will|should|could|would|do|does|did|is|are|was|were)\s+\w+',
                r'\w+\s+(help|please|thanks|now|today|tomorrow|here|there)',
                r'(tell|show|give|find|search|play|stop|start|open|close)\s+\w+',
            ]
            
            for pattern in meaningful_2_word_patterns:
                if re.match(pattern, cleaned_text):
                    return True
            
            # If it's 2 words but doesn't match meaningful patterns, likely not meaningful
            return False
        
        # Single words that are meaningful
        meaningful_single_words = [
            'help', 'stop', 'pause', 'resume', 'continue', 'restart', 'repeat',
            'thanks', 'thank', 'please', 'sorry', 'excuse',
            'weather', 'time', 'date', 'news', 'music', 
            'search', 'find', 'lookup', 'google'
        ]
        
        return cleaned_text in meaningful_single_words
    
    def query_rag(self, text: str) -> tuple[list, bool]:
        self.logger.debug("Querying RAG")
        query_results = self.rag.query(text)
        
        if self.interrupt_count.value > 0:
            return [], True
        
        query_result_scores = [query["score"] for query in query_results] + [0]
        return query_results, max(query_result_scores) >= RAG_CONFIDENCE_THRESHOLD
    
    def perform_websearch(self, text: str, topic: str) -> tuple[list, bool]:
        self.play_search_notification(topic)
        
        websites = self.websearch.ddg_search(topic)
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
        
        return query_results, False
    
    def decide_websearch_needed(self, text: str) -> tuple[bool, str, bool]:
        self.logger.debug("Not enough confident info in RAG, decide search")
        decision, topic = self.llm.decide_websearch(text)
        
        # Check if LLM was interrupted (returns None, None)
        if decision is None or topic is None or self.interrupt_count.value > 0:
            return False, "", True
        
        self.logger.debug(f"Websearch recommended?: {decision} - {topic}")
        return decision == "yes", topic, False
    
    def generate_response(self, text: str, context: str) -> tuple[str, bool]:
        self.logger.debug("Sending to LLM")
        self.osc.send_message("thinking...")
        response = self.llm.send_to_llm(text, context)
        
        # Check if LLM was interrupted (returns None)
        if response is None or self.interrupt_count.value > 0:
            return "", True
        
        self.logger.info(response)
        return response, False
    
    def process_query(self, text: str, continuation: bool, is_goodbye: bool = False) -> tuple[str, bool]:
        current_time = time.time()
        
        # Prevent processing duplicate queries within 2 seconds
        if (text.strip().lower() == self.last_processed_query.strip().lower() and 
            current_time - self.last_processed_time < 2.0 and
            not continuation):
            self.logger.warning(f"Skipping duplicate query: '{text}'")
            return "", False
        
        # Update tracking variables
        self.last_processed_query = text
        self.last_processed_time = current_time
        
        # Clear interrupt context if it has too many entries (prevent buildup)
        if len(self.llm.interrupt_context) > 3:
            self.logger.warning(f"Clearing interrupt context buildup: {len(self.llm.interrupt_context)} entries")
            self.llm.interrupt_context.clear()
        
        # Filter out meaningless short prompts
        if not self._is_meaningful_prompt(text.strip()) and not continuation and not is_goodbye:
            self.logger.info(f"Skipping meaningless short prompt: '{text}'")
            return "", False
        
        if continuation and self.llm.interrupt_context:
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
            response = random.choice(goodbye_responses)
            return response, False
        
        query_results, has_confident_info = self.query_rag(text)
        if self.interrupt_count.value > 0:
            self.interrupt_actions(text, info="querying RAG")
            return "", True
        
        # Only use RAG results if they are confident, otherwise search or use no context
        if has_confident_info:
            self.logger.debug("Using confident RAG information")
            context = ""
            for result in query_results:
                context += result["content"]
        elif ENABLE_WEBSEARCH:
            needs_search, topic, interrupted = self.decide_websearch_needed(text)
            if interrupted:
                self.interrupt_actions(text, info="Decide websearch")
                return "", True
            
            if needs_search:
                search_results, interrupted = self.perform_websearch(text, topic)
                if interrupted:
                    self.interrupt_actions(text, info="Performing websearch")
                    return "", True
                
                # Use fresh search results for context
                context = ""
                for result in search_results:
                    context += result["content"]
            else:
                # No confident RAG info and no search needed - use no context
                context = ""
        else:
            # Web search disabled and no confident RAG info - use no context
            context = ""
        
        response, interrupted = self.generate_response(text, context)
        if interrupted:
            self.interrupt_actions(text, info="LLM Query")
            return "", True
        
        return response, False