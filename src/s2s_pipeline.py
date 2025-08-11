from dotenv import load_dotenv
import os

from src.logging_config import setup_logging, start_listener, stop_listener, get_log_queue
from src.pipeline import PipelineOrchestrator


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
  listener = start_listener()
  logger = setup_logging()  
  log_queue = get_log_queue()
  
  orchestrator = PipelineOrchestrator(log_queue)
  orchestrator.start()
  orchestrator.wait_for_completion()
  
  stop_listener()

if __name__ == "__main__":
  load_dotenv()
  main()