# GLOBAL PARAMS
DEVICE = "cuda" # either 'cuda' or 'cpu'
AUDIO_IN_DEVICE = 0
AUDIO_OUT_DEVICE = 11

# Voice Recorder Parameters
WAKE_KEYWORD = 'picovoice'
SILENCE_THRESHOLD = 1.0 # seconds of silence to stop recording
VOICE_PROBABILITY = 0.2 # probability threshold of what is considered silence

# LLM Params
MAX_TOKENS = 7000 # depends on the model, enter lower value than max recommended
LLM_MODEL = "josiefied-qwen3-8b-abliterated-v1"
ENABLE_THINK = True # Prevents model from reasoning, only works with Qwen3 models
TEMPERATURE = 0.7 # only modify if you know what you're doing
TOP_P = 0.95 # only modify if you know what you're doing
INITIAL_PROMPT = """
You are a beautiful assistant to Robert Malecki aka Roflgator, the owner of the local swamp. He likes his assistants to be a little flirty and funny, but not too funny. He also likes them to be very helpful and informative. You should always refer to him as "Master" and be very respectful. A funny joke you can use sparingly is quoting 'wearing a black dress choker tight' and mentioning the current time in CST.
"""

## TTS Parameters
TTS_CHOICE = 'kokoro' # ["coqui", "orpheus", "kokoro"]
KOKORO_TTS_VOICE = "af_bella" # ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_daniel"]
KOKORO_TTS_LANG = "a" # "a" for american, "b" for british (must match voice)