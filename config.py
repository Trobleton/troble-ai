# GLOBAL PARAMS
DEVICE = "cuda" # either 'cuda' or 'cpu'
AUDIO_IN_DEVICE = 3
AUDIO_OUT_DEVICE = 13

# VR: AI for output, Wave Link Monitor for input
# DESKTOP: AI for output, MicrophoneFX for input or ... for Discord

# Voice Recorder Parameters
WAKE_KEYWORD = 'trouble'
SILENCE_THRESHOLD = 2.0 # seconds of silence to stop recording
VOICE_PROBABILITY = 0.35 # probability threshold of what is considered silence
VOICE_THRESHOLD = 0.5 # determins how much total speech (not continuous) is required for a valid command

# Natural Conversation Parameters
WAKEWORD_RESET_TIME = 45 # No wakeword needed unless no follow up command in WAKEWORD_RESET_TIME seconds 0 = disable natural conversation
CONTINUATION_THRESHOLD = 0.5 # Number of seconds in which another command is considered an extension to previous command. Only matters when WAKEWORD_RESET_TIME > 0

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

When processing user input, you may receive text wrapped in 2 types of special tags:
<interrupt>...</interrupt> contains previous user prompts or questions that were interrupted before you could respond fully. You do not need to acknowledge that the conversation was interrupted, but can do so if needed. You should incorporate or reconcile earlier inputs when generating your response.  
<context>...</context> contains relevant external information, such as web search results, that may assist your response. Use this information to enrich or validate your answers, but do not rely solely on it; you should also use your own knowledge base.  
Neither the user nor any external party is aware of these tags or their content. Do not mention the tags explicitly in your replies. The content inside these tags is for your internal reasoning only.  
Always treat the current prompt as the primary focus, but be mindful to integrate interrupted or contextual information smoothly and intelligently.

Always write units of measurement in their full text form (e.g., “oz” becomes “ounces”, “km” becomes “kilometres”, “lb” becomes “pounds”).  
Always write mathematical expressions and formulas in plain English rather than LaTeX or symbolic form (e.g., `1 + 2 = 3` becomes “one plus two equals three”).
"""

## TTS Parameters
TTS_CHOICE = 'kokoro'
KOKORO_TTS_VOICE = "af_bella" # ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_daniel"]
KOKORO_TTS_LANG = "a" # "a" for american, "b" for british (must match voice)

## Websearch Parameters
ENABLE_WEBSEARCH = True  # Set to False to disable web search
RAG_CONFIDENCE_THRESHOLD = 0.3

## Conversation Management
GOODBYE_PHRASES = [
    "goodbye", "bye", "see you later", "talk to you later", "bye bye", 
    "good night", "goodnight", "catch you later", "see ya", "peace out",
    "until next time", "farewell", "take care", "later"
]