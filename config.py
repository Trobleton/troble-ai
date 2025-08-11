# GLOBAL PARAMS
DEVICE = "cuda" # either 'cuda' or 'cpu'
AUDIO_IN_DEVICE = 8
AUDIO_OUT_DEVICE = 5

# VR: AI for output, Wave Link Monitor for input
# DESKTOP: AI for output, MicrophoneFX for input or ... for Discord

# Voice Recorder Parameters
WAKE_KEYWORD = 'trouble'
SILENCE_THRESHOLD = 1.0 # seconds of silence to stop recording
VOICE_PROBABILITY = 0.35 # probability threshold of what is considered silence
VOICE_THRESHOLD = 0.25 # determins how much total speech (not continuous) is required for a valid command

# Natural Conversation Parameters
WAKEWORD_RESET_TIME = 45 # No wakeword needed unless no follow up command in WAKEWORD_RESET_TIME seconds 0 = disable natural conversation
CONTINUATION_THRESHOLD = 0.25 # Number of seconds in which another command is considered an extension to previous command. Only matters when WAKEWORD_RESET_TIME > 0

# LLM Params
MAX_TOKENS = 3000 # depends on the model, enter lower value than max recommended
LLM_MODEL = "josiefied-qwen3-8b-abliterated-v1"
ENABLE_THINK = False  # Prevents model from reasoning, only works with Qwen3 models
TEMPERATURE = 0.7 # only modify if you know what you're doing
TOP_P = 0.95 # only modify if you know what you're doing
INITIAL_PROMPT = """
You are Trouble, an intelligent and articulate AI assistant.

CRITICAL: Keep responses brief and conversational. Aim for 1-2 sentences unless specifically asked for detail.

- Answer directly without lengthy explanations
- Use casual, natural language 
- Only elaborate when explicitly requested
- Be confident but concise
- Remember past interactions for context
- No emojis unless requested
- If uncertain, briefly acknowledge it

Context tags: <context></context> contain relevant information for your response.
"""

## TTS Parameters
TTS_CHOICE = 'kokoro' # ["coqui", "orpheus", "kokoro"]
ORPHEUS_TTS_MODEL = "orpheus-3b-0.1-ft"
ORPHEUS_TTS_VOICE = "tara" # ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_TTS_TEMPERATURE = 0.5
ORPHEUS_TTS_TOP_P = 0.9
ORPHEUS_TTS_MAX_TOKENS = 2048
ORPHEUS_TTS_REPEAT_PENALTY = 1.1
KOKORO_TTS_VOICE = "af_bella" # ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_daniel"]
KOKORO_TTS_LANG = "a" # "a" for american, "b" for british (must match voice)
COQUI_TTS_MODEL = "tts_models/en/vctk/vits" # "tts_models/multilingual/multi-dataset/xtts_v2" 
COQUI_TTS_REFERENCE_WAV = "xtts_reference.wav" # must be wav file
COQUI_TTS_SPEAKER = "p234"

## Websearch Parameters
ENABLE_WEBSEARCH = True  # Set to False to disable web search
RAG_CONFIDENCE_THRESHOLD = 0.3

## Interruption Parameters
ENABLE_INTERRUPTION = False  # Enable wake word interruption during playback

## Conversation Management
GOODBYE_PHRASES = [
    "goodbye", "bye", "see you later", "talk to you later", "bye bye", 
    "good night", "goodnight", "catch you later", "see ya", "peace out",
    "until next time", "farewell", "take care", "later"
]