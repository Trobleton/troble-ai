import argparse
from dotenv import load_dotenv
import os
import io
import logging
import time

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
  play_wav_file(output_buffer, device=AUDIO_OUT_DEVICE)

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
        tts = TTSOrpheus()

    vrchat_osc = VRChatOSC()
    ask_wakeword = True
    last_command_time = 0

    while True:
        if (time.time() - last_command_time) > WAKEWORD_RESET_TIME:
            ask_wakeword = True

        if ask_wakeword:
            logger.debug("Listening for wake word...")
            audio_recorder.record_wake_word()
            last_command_time = time.time()
            ask_wakeword = False
            
        logger.debug("Listening for command...")
        command_buffer, command_duration = audio_recorder.record_command()

        command_buffer.seek(0, io.SEEK_END)
        command_size = command_buffer.tell()
        command_buffer.seek(0)

        num_samples = command_size // 2  # 16-bit audio
        if num_samples < audio_recorder.porcupine.sample_rate * (VOICE_THRESHOLD + SILENCE_THRESHOLD):
            logger.debug("Command too short, skipping")

        output_filename = "command.wav"
        save_wav_file(command_buffer, output_filename)

        logger.debug("Running Speech-to-Text...")
        command_buffer.seek(0)
        text_segments = whisper.transcribe(command_buffer)
        text = ", ".join([segment.text for segment in text_segments])
        logger.info(text)

        if not text:
            logger.debug("No command detected")
            continue

        if not ask_wakeword:
            last_command_time = time.time()

        decision, topic = llm.decide_websearch(text)
        logger.debug(f"Decision: {decision}, Topic: {topic}")
        context = ""
        query_results = []

        if decision == "yes":
            logger.debug(f"querying RAG")
            query_results = rag.query(topic)
            query_result_scores = [query["score"] for query in query_results] + [0]

            if (max(query_result_scores) < RAG_CONFIDENCE_THRESHOLD):
                logger.debug(f"Not enough confidence in RAG results, requires search")
                speech_thread = threading.Thread(target=searching_speech_worker, args=(tts, f"Searching the web for {topic}"))
                speech_thread.start()

                websites = websearch.ddg_search(text)
                logger.debug(f"fetching website contents")
                web_contents = websearch.fetch_content(websites)

                logger.debug(f"adding contents to RAG")
                for document in web_contents:
                    rag.add_document(document)

                logger.debug(f"querying RAG again")
                query_results = rag.query(topic)
                logger.info(query_results)

                speech_thread.join()
        
        logger.debug(f"Info in RAG exists, no search needed")

        for result in query_results:
            context += result["content"]

        logger.debug("sending to llm...")
        response = llm.send_to_llm(text, context)
        logger.info(response)

        logger.debug("Synthesizing speech...")
        output_buffer, output_duration = tts.synthesize(response)

        output_buffer.seek(0)
        output_filename = "output.wav"
        logger.debug(f"Saving output to {output_filename}")
        save_wav_file(output_buffer, output_filename)
        output_buffer.seek(0)

        logger.debug("Playing output...")
        play_wav_file(output_filename)

        # TODO: Show output in VRChat Text Box. Char. limit is 144, so replace text during duration to fit

        output_buffer.seek(0)

if __name__ == "__main__":
    load_dotenv()
    main()

