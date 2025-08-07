import logging
import wave
import array
import threading
from pvspeaker import PvSpeaker

logger = logging.getLogger("speech_to_speech.s2s_pipeline")

def save_wav_file(wav_bytes, wav_filename):
    with wave.open(wav_bytes, 'rb') as in_wav:
        nchannels = in_wav.getnchannels()
        sampwidth = in_wav.getsampwidth()
        framerate = in_wav.getframerate()
        nframes = in_wav.getnframes()
        comptype = in_wav.getcomptype()
        compname = in_wav.getcompname()

        frames = in_wav.readframes(nframes)

        with wave.open(wav_filename, 'wb') as out_wav:
            out_wav.setnchannels(nchannels)
            out_wav.setsampwidth(sampwidth)
            out_wav.setframerate(framerate)
            out_wav.setcomptype(comptype, compname)

            out_wav.writeframes(frames)
        
        logger.debug(f"Audio successfully saved to {wav_filename}")

def play_wav_file(wav_bytes, device=-1):
    def blocking_call(speaker):
        speaker.flush()
    
    def worker_function(speaker, completion_event):
        blocking_call(speaker)
        completion_event.set()

    def split_list(input_list, x):
        return [input_list[i:i + x] for i in range(0, len(input_list), x)]
    
    wav_file = wave.open(wav_bytes, 'rb')
    sample_rate = wav_file.getframerate()
    bits_per_sample = wav_file.getsampwidth() * 8
    num_channels = wav_file.getnchannels()
    num_samples = wav_file.getnframes()

    if bits_per_sample not in [8, 16, 24, 32]:
        logger.error(f"Unsupported bits per sample: {bits_per_sample}")
        wav_file.close()
        exit()
    
    if num_channels != 1:
        logger.error(f"WAV file must have a single channel (MONO)")
        wav_file.close()
        exit()

    speaker = PvSpeaker(sample_rate=sample_rate, bits_per_sample=bits_per_sample, buffer_size_secs=20, device_index=device)
    print("Using device: %s" % speaker.selected_device)

    wav_bytes = wav_file.readframes(num_samples)

    pcm = None
    if bits_per_sample == 8:
        pcm = array.array('B', wav_bytes)
    elif bits_per_sample == 16:
        pcm = list(array.array('h', wav_bytes))
    elif bits_per_sample == 24:
        pcm = []
        for i in range(0, len(wav_bytes), 3):
            sample = int.from_bytes(wav_bytes[i:i+3], byteorder='little', signed=True)
            pcm.append(sample)
    elif bits_per_sample == 32:
        pcm = list(array.array('i', wav_bytes))

    pcm_list = split_list(pcm, sample_rate)
    speaker.start()

    print("Playing audio...")
    for pcm_sublist in pcm_list:
        sublsit_length = len(pcm_sublist)
        total_written_length = 0
        while total_written_length < sublsit_length:
            written_length = speaker.write(pcm_sublist[total_written_length:])
            total_written_length += written_length
    
    logger.debug("Waiting for audio to finish playing...")

    completion_event = threading.Event()
    worker_thread = threading.Thread(target=worker_function, args=(speaker, completion_event))
    worker_thread.start()
    completion_event.wait()
    worker_thread.join()

    speaker.stop()

    logger.debug("Finished playing audio...")
    wav_file.close()

def play_wav_file_interruptible(wav_bytes, device=-1, interrupt_event=None):
    """Play audio with ability to interrupt via threading.Event"""
    def blocking_call(speaker):
        speaker.flush()
    
    def worker_function(speaker, completion_event):
        blocking_call(speaker)
        completion_event.set()

    def split_list(input_list, x):
        return [input_list[i:i + x] for i in range(0, len(input_list), x)]
    
    wav_file = wave.open(wav_bytes, 'rb')
    sample_rate = wav_file.getframerate()
    bits_per_sample = wav_file.getsampwidth() * 8
    num_channels = wav_file.getnchannels()
    num_samples = wav_file.getnframes()
    
    # Calculate expected duration
    expected_duration = num_samples / sample_rate
    logger.debug(f"Audio info: {num_samples} samples, {sample_rate}Hz, {expected_duration:.2f}s duration")

    if bits_per_sample not in [8, 16, 24, 32]:
        logger.error(f"Unsupported bits per sample: {bits_per_sample}")
        wav_file.close()
        return False
    
    if num_channels != 1:
        logger.error(f"WAV file must have a single channel (MONO)")
        wav_file.close()
        return False
    
    if num_samples == 0:
        logger.error("Audio file has no samples")
        wav_file.close()
        return False

    speaker = PvSpeaker(sample_rate=sample_rate, bits_per_sample=bits_per_sample, buffer_size_secs=20, device_index=device)
    print("Using device: %s" % speaker.selected_device)

    wav_bytes = wav_file.readframes(num_samples)

    pcm = None
    if bits_per_sample == 8:
        pcm = array.array('B', wav_bytes)
    elif bits_per_sample == 16:
        pcm = list(array.array('h', wav_bytes))
    elif bits_per_sample == 24:
        pcm = []
        for i in range(0, len(wav_bytes), 3):
            sample = int.from_bytes(wav_bytes[i:i+3], byteorder='little', signed=True)
            pcm.append(sample)
    elif bits_per_sample == 32:
        pcm = list(array.array('i', wav_bytes))

    pcm_list = split_list(pcm, sample_rate)
    speaker.start()

    print("Playing audio...")
    interrupted = False
    
    import time
    start_time = time.time()
    
    for pcm_sublist in pcm_list:
        if interrupt_event and interrupt_event.is_set():
            interrupted = True
            break
            
        sublsit_length = len(pcm_sublist)
        total_written_length = 0
        while total_written_length < sublsit_length:
            if interrupt_event and interrupt_event.is_set():
                interrupted = True
                break
            written_length = speaker.write(pcm_sublist[total_written_length:])
            total_written_length += written_length
        
        if interrupted:
            break
    
    if interrupted:
        actual_duration = time.time() - start_time
        logger.debug(f"Audio playbook interrupted during main loop after {actual_duration:.2f}s")
        speaker.stop()
        wav_file.close()
        return False
    
    logger.debug("Main audio loop completed, waiting for flush...")

    completion_event = threading.Event()
    worker_thread = threading.Thread(target=worker_function, args=(speaker, completion_event))
    worker_thread.start()
    
    while not completion_event.is_set():
        if interrupt_event and interrupt_event.is_set():
            interrupted = True
            logger.debug("Interrupted during flush wait")
            break
        completion_event.wait(timeout=0.1)
    
    if interrupted:
        actual_duration = time.time() - start_time
        logger.debug(f"Audio playback interrupted during flush after {actual_duration:.2f}s")
        speaker.stop()
        wav_file.close()
        return False
    
    worker_thread.join()
    speaker.stop()

    actual_duration = time.time() - start_time
    logger.debug(f"Audio playback completed successfully. Expected: {expected_duration:.2f}s, Actual: {actual_duration:.2f}s")
    
    # Check if playback ended significantly early (more than 0.5 seconds difference)
    if expected_duration - actual_duration > 0.5:
        logger.warning(f"Audio ended {expected_duration - actual_duration:.2f}s early - possible audio truncation")
    
    wav_file.close()
    return True