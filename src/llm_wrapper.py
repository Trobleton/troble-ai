from openai import OpenAI
import logging
import json
import re
import os
from config import *
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass


class LLMWrapper():
  def __init__(self, interrupt_count:SynchronizedClass ):
    self.logger = logging.getLogger("speech_to_speech.llm_wrapper")
    self.interrupt_count = interrupt_count
    self.interrupt_context = []

    self.api = os.getenv("OPENAI_API")
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.model = LLM_MODEL
    self.global_chat_history = []
    self.current_chat_history = []
    self.current_chat_history_length = 0
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.chat_history_path = os.path.join(project_root_dir, "data", "chat_history.json")
    self.max_tokens = int(MAX_TOKENS * 0.75)
    
    self.initial_prompt = INITIAL_PROMPT
    self.initial_prompt += "Do not style your response using markdown formatting. Note that you responses must be in a conversation format, thus not text fomatting is allowed to make the output look nice after it has been rendered."
    if TTS_CHOICE == "orpheus":
      self.initial_prompt += " Also, add paralinguistic elements like <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp> or uhm for more human-like speech whenever it fits, but do not overdo it, please only add it when necessary and not often."
    if TTS_CHOICE == "kokoro":
      self.initial_prompt += "Here are some rules regarding how the output should be formatted such that it could work with text-to-speech. 1. To adjust intonation, try punctuation ;:,.!?—…\"()“” or stress ˈ and ˌ"
    self.initial_prompt = self.initial_prompt.replace("\n", "")
    self.initial_prompt_length = len(self.initial_prompt.split(" "))
    
    self.websearch_classifier_prompt = """
    You are a classifier that determines whether a user’s request requires an external web search.

    Rules for deciding:
    - Answer "yes" if the prompt is about facts, knowledge, history, current events, or time-sensitive information
    - Answer "no" if the request can be answered without external knowledge (general conversation, opinions, jokes, instructions, etc.).
    - If answering "yes", provide the topic to be searched for in no more than 5 words.  
    - If answering "no", the main topic should be "None".
    - Correct the spelling of words in the main topic if necessary as the input prompt it the output of speech-to-text, and the text will be slightly off.

    Output format (no extra words, no punctuation except as shown):  
    `<yes/no>+-+<main topic>`

    Examples:
    - "How tall is the Empire state?" → `yes+-+Empire State Height`
    - "What is the price of Bitcoin currently?" → `yes+-+Bitcoin Price`
    - "How are you doing today?" → `no+-+None`
    - "Explain what a black hole is." → `yes+-+What is Black Hole`
    - "Write me a poem about cats." → `no+-+None`
    - "Tell me the weather in Tokyo." → `yes+-+Tokyo Weather`
    """
    
    self.client = OpenAI(base_url=self.api, api_key=self.api_key)
    
    self._load_convo_history()
    
  def _load_convo_history(self):
    self.logger.debug("Loading conversation history")

    if not os.path.exists(self.chat_history_path):
      self.logger.debug("Chat history file doesn't exist, creating empty history")
      os.makedirs(os.path.dirname(self.chat_history_path), exist_ok=True)
      with open(self.chat_history_path, 'w', encoding="utf-8") as f:
        json.dump({"history": []}, f)
      self.global_chat_history = []
      return

    chat_history_file = open(self.chat_history_path, 'r', encoding="utf-8")
    self.global_chat_history = json.load(chat_history_file)["history"]
    chat_history_file.close()
    
    index = len(self.global_chat_history) - 1
    while index > -1 and ((self.current_chat_history_length + self.initial_prompt_length) < self.max_tokens ):
      cur_message = self.global_chat_history[index]
      self.current_chat_history.insert(0, cur_message["message"])
      self.current_chat_history_length += cur_message["length"]
      index -= 1

  def _write_chat_history(self):
    self.logger.debug(f"Saving conversation history")

    chat_history_file = open(self.chat_history_path, 'w', encoding="utf-8")
    json.dump({
        "history" : self.global_chat_history
      },
      chat_history_file
    )
    chat_history_file.close()

  def _filter_think(self, text):
    marker = "</think>"
    index = text.find(marker)
    
    if index != -1:
      # Found the marker, return text after it
      filtered_text = text[index + len(marker):].replace("\n", "")
    else:
      # No marker found, return original text
      filtered_text = text.replace("\n", "")
    
    return filtered_text

  def _filter_markdown(self, text):
    # Remove code blocks
    filtered_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code
    filtered_text = re.sub(r"`([^`]*)`", r"\1", filtered_text)
    # Remove bold/italic markers
    filtered_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", filtered_text)
    filtered_text = re.sub(r"(\*|_)(.*?)\1", r"\2", filtered_text)
    # Remove headers
    filtered_text = re.sub(r"^#+\s*", "", filtered_text, flags=re.MULTILINE)
    # Remove blockquotes
    filtered_text = re.sub(r"^>\s*", "", filtered_text, flags=re.MULTILINE)
    # Remove horizontal rules
    filtered_text = re.sub(r"^-{3,}", "", filtered_text, flags=re.MULTILINE)

    return filtered_text.strip()
  
  def _filter_emoji(self, text):
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      "]+", flags=re.UNICODE
    )

    filtered_text = emoji_pattern.sub(r'', text)
    
    return filtered_text
  
  def _filter_expressions(self, text):
    """Remove TTS expressions like <smile>, <laugh>, etc."""
    # Pattern to match expressions in angle brackets
    expression_pattern = re.compile(r'<[^>]*>', re.IGNORECASE)
    filtered_text = expression_pattern.sub('', text)
    
    # Clean up any double spaces that might result from removing expressions
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    
    return filtered_text

  def decide_websearch(self, text):
    prompt_messages = [
      {"role": "system", "content": self.websearch_classifier_prompt},
      {"role": "user", "content": text + "/no_think"}
    ]
    
    response_text = ""
    
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      stream=True
      # max_tokens=150,
    )

    for chunk in stream:
      if self.interrupt_count.value > 0:
        # On interrupt
        return None, None
      
      data = chunk.choices[0].delta.content
      if data is not None:
        response_text += data

    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
    # Remove expressions like <smile>, <laugh> when using Kokoro TTS
    if TTS_CHOICE == "kokoro":
      response_text = self._filter_expressions(response_text)
    
    # Handle cases where the LLM doesn't follow the expected format
    if "+-+" in response_text:
      require_search, topic = response_text.split("+-+")
    else:
      # Fallback: try to parse manually or default to no search
      self.logger.warning(f"LLM response doesn't contain expected separator '+-+': '{response_text}'")
      response_lower = response_text.lower().strip()
      if response_lower.startswith("yes"):
        require_search = "yes"
        # Try to extract topic after "yes"
        topic_match = response_text.lower().replace("yes", "").strip()
        topic = topic_match if topic_match else "general search"
      elif response_lower.startswith("no"):
        require_search = "no"
        topic = "none"
      else:
        # Default to no search if we can't parse
        require_search = "no" 
        topic = "none"

    return require_search.lower(), topic.lower()

  def send_to_llm(self, text, context = ""):
    if not ENABLE_THINK:
      text = text + " /no_think" # disable reasoning

    text_length = len(text.split(" "))
    
    prompt_modification = ""
    # if interrupted, let LLM know
    if len(self.interrupt_context) > 0:
      for entry in self.interrupt_context:
        prompt_modification += "<interrupt>"+ entry + "</interrupt>\n"
        
    interrupt_text = prompt_modification + text
    interrupt_text_length = len(interrupt_text.replace("\n", " ").split(" "))
    
    self.current_chat_history.append(
      {"role": "user", "content": interrupt_text}
    )
    
    self.current_chat_history_length += interrupt_text_length
    
    while self.current_chat_history and ((self.current_chat_history_length + self.initial_prompt_length) >= self.max_tokens ):
      removed_chat_length = len(self.current_chat_history.pop(0).split(" "))
      self.current_chat_history_length -= removed_chat_length
    
    prompt_messages = [{"role": "system", "content": self.initial_prompt}]
    prompt_messages.extend(self.current_chat_history)
    
    # if context exists, add to user prompt
    if context != "":
      prompt_modification = "<context>"+ context.replace("\n", "") + "</context>\n\n"
    
    prompt_messages[-1]["content"] =  prompt_modification + prompt_messages[-1]["content"]
    
    # Log the complete prompt being sent to LLM
    self.logger.info(f"PROMPT: {prompt_messages[-1]['content']}")
    
    response_text = ""
    
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      stream=True
      # max_tokens=150,
    )
    for chunk in stream:
      if self.interrupt_count.value > 0:
        # On interrupt
        return None

      data = chunk.choices[0].delta.content
      if data is not None:
        response_text += data
    
    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
    # Remove expressions like <smile>, <laugh> when using Kokoro TTS
    if TTS_CHOICE == "kokoro":
      response_text = self._filter_expressions(response_text)
    
    # Log the generated response
    self.logger.info(f"RESPONSE: {response_text}")
    
    response_length = len(response_text.split(" "))
    
    self.global_chat_history.append({
      "message": {"role": "user", "content": interrupt_text},
      "length": interrupt_text_length
    })
    self.global_chat_history.append({
      "message": {"role": "assistant", "content": response_text},
      "length": response_length
    })
    self.current_chat_history.append(
      {"role": "assistant", "content": response_text}
      )
    
    self._write_chat_history()
    
    self.logger.debug("Response returned")

    return response_text
    
    
    
    