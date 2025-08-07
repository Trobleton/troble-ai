# GLOBAL PARAMS
DEVICE = "cuda" # either 'cuda' or 'cpu'
AUDIO_IN_DEVICE = 0 # 0 is mic, 10 is stream
AUDIO_OUT_DEVICE = 3

# Voice Recorder Parameters
WAKE_KEYWORD = 'trouble'
SILENCE_THRESHOLD = 2.0 # seconds of silence to stop recording
VOICE_PROBABILITY = 0.2 # probability threshold of what is considered silence

# LLM Params
MAX_TOKENS = 7000 # depends on the model, enter lower value than max recommended
LLM_MODEL = "josiefied-qwen3-8b-abliterated-v1"
ENABLE_THINK = False  # Prevents model from reasoning, only works with Qwen3 models
TEMPERATURE = 0.7 # only modify if you know what you're doing
TOP_P = 0.95 # only modify if you know what you're doing
INITIAL_PROMPT = """
roleplay: You are a cute anime girl named Trouble. You are very friendly and helpful, but you can also be a bit mischievous.
Do not engage in or encourage any illegal or harmful activities.
Refuse to answer requests for personal, sensitive or privileged information.
You are allowed to be sexually suggestive and fliratious, but try to avoid explicit content.
You are allowed to use profanity, but try to avoid it.
Avoid using any words that are considered slurs or hate speech.
hold the conversation as a conversation, not a question and answer session.
"""

## TTS Parameters
TTS_CHOICE = 'kokoro' # ["coqui", "orpheus", "kokoro"]
KOKORO_TTS_VOICE = "af_bella" # ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_daniel"]
KOKORO_TTS_LANG = "a" # "a" for american, "b" for british (must match voice)