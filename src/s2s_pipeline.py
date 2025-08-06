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
from src.tts import TTS
from src.utils import save_wav_file, play_wav_file
from src.osc import VRChatOSC
from config import *

import sys
import os
from typing import Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger("speech_to_speech.s2s_pipeline")

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
    tts = TTS()

    vrchat_osc = VRChatOSC()
    conversation_mode = False  # Set to True to keep listening after response

    while True:
        if not conversation_mode:
            logger.debug("Listening for wake word...")
            audio_recorder.record_wake_word()
            silence_threshold = SILENCE_THRESHOLD  # Use your normal threshold
        else:
            silence_threshold = 3  # 3 seconds for conversation mode

        logger.debug("Listening for command...")
        vrchat_osc.send_message("(listening)")
        text, _ = listen_for_command(audio_recorder, whisper, silence_threshold=silence_threshold)
        if not text:
            if conversation_mode:
                logger.debug("No command detected in conversation mode, exiting to wake word mode.")
                conversation_mode = False
                vrchat_osc.clear_message()
            continue

        vrchat_osc.send_message("(thinking)")
        logger.debug("Sending to llm...")
        response = llm.send_to_llm(text)
        logger.info(response)

        logger.debug("Synthesizing response...")
        output_buffer, output_duration = tts.synthesize(response)

        output_buffer.seek(0)
        output_filename = "output.wav"
        logger.debug("Saving wav file...")
        save_wav_file(output_buffer, output_filename)
        output_buffer.seek(0)

        logger.debug("Playing response...")
        vrchat_osc.clear_message()
        play_wav_file(output_buffer, device=AUDIO_OUT_DEVICE)
        output_buffer.seek(0)

        logger.debug("Response completed, listening for next command...")
        conversation_mode = True

if __name__ == "__main__":
    load_dotenv()
    main()

