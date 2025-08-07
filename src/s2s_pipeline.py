import argparse
from dotenv import load_dotenv
import os
import io
import logging

from src.logging_config import setup_logging
setup_logging()

from src.recorder import Recorder
from src.stt import STT
from src.llm_wrapper import LLMWrapper
from src.tts_kokoro import TTSKokoro
from src.tts_orpheus import TTSOrpheus
from src.utils import save_wav_file, play_wav_file, play_wav_file_interruptible
from src.osc import VRChatOSC
from src.web_search import WebSearcher
from src.rag_langchain import RAGLangchain
from config import *

import os
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger("speech_to_speech.s2s_pipeline")

def searching_speech_worker(tts, text):
  output_buffer, output_duration = tts.synthesize(text)
  output_buffer.seek(0)
  play_wav_file(output_buffer)

def listen_for_command(
    audio_recorder: Recorder,
    whisper: STT,
    silence_threshold: float = SILENCE_THRESHOLD
):
    command_buffer = audio_recorder.record_command(silence_threshold)

    command_buffer.seek(0, io.SEEK_END)
    command_size = command_buffer.tell()
    command_buffer.seek(0)

    num_samples = command_size // 2
    if num_samples < audio_recorder.porcupine.sample_rate * 2:
        logger.debug("No speech detected")
        return None, None

    command_filename = "command.wav"
    command_buffer.seek(0)
    text_segments = whisper.transcribe(command_buffer)
    text = "\n".join([segment.text for segment in text_segments])
    logger.info(text)
    return text, command_filename

def main():
    audio_recorder = Recorder(os.getenv("PICOVOICE_API_KEY"))
    whisper = STT(vad_active=True, device=DEVICE)
    llm = LLMWrapper(
        api=os.getenv("OPENAI_API"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    websearch = WebSearcher()
    rag = RAGLangchain()
    
    if TTS_CHOICE == "kokoro":
        tts = TTSKokoro()
    elif TTS_CHOICE == "orpheus":
        tts = TTSOrpheus(
            api=os.getenv("OPENAI_API"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    vrchat_osc = VRChatOSC()
    conversation_mode = False  # Set to True to keep listening after response
    pending_text = None  # Store text from brief listening period

    while True:
        import time
        pipeline_start = time.time()
        
        # Check if we have pending text from brief listening period
        if pending_text:
            text = pending_text
            pending_text = None
            logger.debug("Processing pending command from brief listening period")
        else:
            if not conversation_mode:
                vrchat_osc.clear_message()
                logger.debug("Listening for wake word...")
                audio_recorder.record_wake_word()
                silence_threshold = SILENCE_THRESHOLD  # Use your normal threshold
            else:
                silence_threshold = SILENCE_THRESHOLD + 1  # Slightly longer for conversation mode

            logger.debug("Listening for command...")
            vrchat_osc.send_message("(listening)")
            stt_start = time.time()
            text, _ = listen_for_command(audio_recorder, whisper, silence_threshold=silence_threshold)
            stt_time = time.time() - stt_start
            logger.debug(f"STT took {stt_time:.2f}s")
            
            if not text:
                if conversation_mode:
                    logger.debug("No command detected in conversation mode, exiting to wake word mode.")
                    conversation_mode = False
                    vrchat_osc.clear_message()
                continue
        
        rag_start = time.time()
        decision, topic = llm.decide_websearch(text)
        decision_time = time.time() - rag_start
        logger.debug(f"LLM decision took {decision_time:.2f}s. Websearch recommended?: {decision} - {topic}")
        
        context = ""
        query_results = []
        query_result_scores = [0]  # Initialize with default value

        if decision == "yes":
            logger.debug(f"Querying RAG")
            query_results = rag.query(topic)
            # add extra 0 in case RAG is empty and returns empty list
            query_result_scores = [query["score"] for query in query_results] + [0]

            if  max(query_result_scores) < RAG_CONFIDENCE_THRESHOLD:
                logger.debug(f"Not enough confident info in RAG, requires search")
                
                # Play search speech while web search occurs
                speech_thread = threading.Thread(target=searching_speech_worker, args=(tts, f"Searching the web for {topic}"))
                speech_thread.start()
                
                search_start = time.time()
                websites = websearch.ddg_search(text)
                logger.debug(f"Fetching website contents")
                web_contents = websearch.fetch_content(websites)
                search_time = time.time() - search_start
                logger.debug(f"Web search took {search_time:.2f}s")

                logger.debug(f"Adding contents to RAG")
                for document in web_contents:
                    rag.add_document(document)

                logger.debug(f"Querying RAG Again")
                query_results = rag.query(topic)
                logger.info(query_results)

                # Ensure search speech completes before continuing
                logger.debug("Waiting for search speech to complete...")
                speech_thread.join()
                logger.debug("Search speech completed")
                
                # Small delay to ensure audio device is fully released
                import time
                time.sleep(0.2)
            else:
                logger.debug(f"Info in RAG exists, no search needed")
        
        rag_time = time.time() - rag_start
        logger.debug(f"Total RAG/search phase took {rag_time:.2f}s")
        
        for result in query_results:
            context += result["content"]

        vrchat_osc.send_message("(thinking)")
        logger.debug("Sending to llm...")
        llm_start = time.time()
        response = llm.send_to_llm(text)
        llm_time = time.time() - llm_start
        logger.debug(f"LLM response took {llm_time:.2f}s")
        logger.info(response)

        logger.debug("Synthesizing response...")
        tts_start = time.time()
        output_buffer, output_duration = tts.synthesize(response)
        tts_time = time.time() - tts_start
        logger.debug(f"TTS took {tts_time:.2f}s")

        output_buffer.seek(0)
        output_filename = "output.wav"
        logger.debug("Saving wav file...")
        save_wav_file(output_buffer, output_filename)
        output_buffer.seek(0)
        
        total_time = time.time() - pipeline_start
        logger.info(f"PIPELINE TIMING - Total: {total_time:.2f}s | STT: {stt_time:.2f}s | LLM: {llm_time:.2f}s | TTS: {tts_time:.2f}s | RAG: {rag_time:.2f}s")

        logger.debug("Playing response...")
        vrchat_osc.clear_message()
        
        # Create interrupt event for wake word detection during playback
        interrupt_event = threading.Event()
        
        # Start wake word detection in separate thread during playback
        wake_word_thread = threading.Thread(
            target=audio_recorder.listen_for_wake_word_with_interrupt, 
            args=(interrupt_event,)
        )
        wake_word_thread.start()
        
        # Play audio with interruption capability
        logger.debug("Starting interruptible playback...")
        playback_completed = play_wav_file_interruptible(
            output_buffer, 
            device=AUDIO_OUT_DEVICE, 
            interrupt_event=interrupt_event
        )
        
        # Signal wake word thread to stop if audio completed normally
        if playback_completed:
            interrupt_event.set()
        
        # Wait for wake word thread to finish
        logger.debug("Waiting for wake word thread to join...")
        wake_word_thread.join()
        logger.debug("Wake word thread joined")
        
        output_buffer.seek(0)
        
        logger.debug(f"About to check if playback_completed ({playback_completed}) is True")
        logger.debug(f"playback_completed == True: {playback_completed == True}")
        logger.debug(f"bool(playback_completed): {bool(playback_completed)}")
        
        if playback_completed:
            logger.debug("Entering playback completed branch")
            logger.debug("Response completed, brief listening for next command...")
            # Listen briefly for next command after audio completion
            vrchat_osc.send_message("(listening)")
            brief_silence_threshold = SILENCE_THRESHOLD
            text, _ = listen_for_command(audio_recorder, whisper, silence_threshold=brief_silence_threshold)
            vrchat_osc.clear_message()
            
            if text:
                logger.debug("Command detected during brief listening period")
                conversation_mode = True
                pending_text = text  # Store for next loop iteration
            else:
                logger.debug("No command during brief listening, returning to wake word mode")
                conversation_mode = False
        else:
            logger.debug("Entering playback interrupted branch")
            logger.debug("Response interrupted by wake word, listening for command immediately...")
            conversation_mode = True  # Stay in conversation mode to skip wake word detection

if __name__ == "__main__":
    load_dotenv()
    main()

