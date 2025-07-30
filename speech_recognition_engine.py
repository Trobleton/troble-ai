import whisper
import numpy as np
import torch
import threading
import queue
import time
import tempfile
import os
from typing import Optional, Callable, Dict, List
from audio_device_manager import AudioDeviceManager
from audio_engine import AudioEngine


class SpeechRecognitionEngine:
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize the Speech Recognition Engine with Whisper

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ("cuda" or "cpu", auto-detect if None)
            language: Target language (None for auto-detection)
        """
        self.model_size = model_size
        self.language = language
        self.model = None

        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Processing state
        self.is_processing = False
        self.processing_queue = queue.Queue()
        self.processing_thread = None

        # Callbacks
        self.on_transcription_start = None
        self.on_transcription_complete = None
        self.on_transcription_error = None

        # Performance tracking
        self.transcription_times = []
        self.total_transcriptions = 0

        print(f"🤖 Speech Recognition Engine initialized")
        print(f"   Model: whisper-{model_size}")
        print(f"   Device: {self.device}")
        print(f"   Language: {language or 'auto-detect'}")

    def load_model(self) -> bool:
        """Load the Whisper model"""
        try:
            print(f"📥 Loading Whisper model '{self.model_size}'...")
            start_time = time.time()

            self.model = whisper.load_model(self.model_size, device=self.device)

            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")

            # Test the model with a small dummy input
            self._test_model()

            return True

        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False

    def _test_model(self):
        """Test the model with a dummy input to ensure it's working"""
        try:
            print("🧪 Testing model with dummy input...")

            # Create a short dummy audio (1 second of silence)
            dummy_audio = np.zeros(16000, dtype=np.float32)

            result = self.model.transcribe(dummy_audio, language=self.language)
            print("✅ Model test completed")

        except Exception as e:
            print(f"⚠️ Model test failed: {e}")

    def set_callbacks(
        self,
        on_transcription_start: Optional[Callable] = None,
        on_transcription_complete: Optional[Callable[[str, Dict], None]] = None,
        on_transcription_error: Optional[Callable[[str], None]] = None,
    ):
        """Set callback functions for transcription events"""
        self.on_transcription_start = on_transcription_start
        self.on_transcription_complete = on_transcription_complete
        self.on_transcription_error = on_transcription_error

    def transcribe_audio(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Dict:
        """
        Transcribe audio data to text

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            start_time = time.time()

            if self.on_transcription_start:
                self.on_transcription_start()

            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                task="transcribe",  # vs "translate"
                fp16=self.device == "cuda",  # Use FP16 on GPU for speed
                verbose=False,
            )

            transcription_time = time.time() - start_time
            self.transcription_times.append(transcription_time)
            self.total_transcriptions += 1

            # Extract key information
            transcription_result = {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result["segments"],
                "processing_time": transcription_time,
                "audio_duration": len(audio_data) / 16000,
                "confidence": self._calculate_average_confidence(result["segments"]),
            }

            print(f"📝 Transcription completed in {transcription_time:.2f}s")
            print(f'🗣️ Text: "{transcription_result["text"]}"')
            print(f"🌍 Language: {transcription_result['language']}")
            print(f"📊 Confidence: {transcription_result['confidence']:.2f}")

            if self.on_transcription_complete:
                self.on_transcription_complete(
                    transcription_result["text"], transcription_result
                )

            return transcription_result

        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"❌ {error_msg}")

            if self.on_transcription_error:
                self.on_transcription_error(error_msg)

            raise e

    def _calculate_average_confidence(self, segments: List[Dict]) -> float:
        """Calculate average confidence from segments"""
        if not segments:
            return 0.0

        # Whisper doesn't always provide confidence scores
        # This is a placeholder for when they do
        confidences = []
        for segment in segments:
            if "confidence" in segment:
                confidences.append(segment["confidence"])

        if confidences:
            return sum(confidences) / len(confidences)
        else:
            # Estimate confidence based on other factors
            return 0.8  # Default reasonable confidence

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Simple resampling (for basic use - librosa would be better for production)"""
        if orig_sr == target_sr:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(np.float32)

    def transcribe_file(self, audio_file_path: str) -> Dict:
        """Transcribe an audio file"""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        print(f"📁 Transcribing file: {audio_file_path}")

        try:
            result = self.model.transcribe(
                audio_file_path,
                language=self.language,
                task="transcribe",
                fp16=self.device == "cuda",
                verbose=True,
            )

            transcription_result = {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result["segments"],
                "file_path": audio_file_path,
            }

            return transcription_result

        except Exception as e:
            print(f"❌ File transcription failed: {e}")
            raise e

    def start_async_processing(self):
        """Start background thread for async transcription processing"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            print("⚠️ Processing thread already running")
            return

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("🔄 Async processing thread started")

    def stop_async_processing(self):
        """Stop background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        print("⏹️ Async processing thread stopped")

    def queue_for_transcription(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Queue audio data for async transcription"""
        if not self.is_processing:
            self.start_async_processing()

        self.processing_queue.put((audio_data.copy(), sample_rate, time.time()))
        print(
            f"📋 Audio queued for transcription (queue size: {self.processing_queue.qsize()})"
        )

    def _processing_worker(self):
        """Background worker thread for processing transcription queue"""
        print("🔄 Transcription processing worker started")

        while self.is_processing:
            try:
                # Get audio from queue with timeout
                audio_data, sample_rate, queue_time = self.processing_queue.get(
                    timeout=1.0
                )

                queue_wait_time = time.time() - queue_time
                if queue_wait_time > 0.1:
                    print(f"⏱️ Queue wait time: {queue_wait_time:.2f}s")

                # Process the audio
                self.transcribe_audio(audio_data, sample_rate)

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing worker error: {e}")
                continue

        print("🔄 Transcription processing worker stopped")

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.transcription_times:
            return {
                "total_transcriptions": 0,
                "average_processing_time": 0,
                "fastest_processing_time": 0,
                "slowest_processing_time": 0,
            }

        return {
            "total_transcriptions": self.total_transcriptions,
            "average_processing_time": np.mean(self.transcription_times),
            "fastest_processing_time": np.min(self.transcription_times),
            "slowest_processing_time": np.max(self.transcription_times),
            "queue_size": self.processing_queue.qsize()
            if hasattr(self, "processing_queue")
            else 0,
        }

    def change_model(self, new_model_size: str) -> bool:
        """Change the Whisper model size"""
        if new_model_size == self.model_size:
            print(f"⚠️ Already using model '{new_model_size}'")
            return True

        print(f"🔄 Changing model from '{self.model_size}' to '{new_model_size}'")

        old_model_size = self.model_size
        self.model_size = new_model_size

        # Clear old model from memory
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Load new model
        if self.load_model():
            print(f"✅ Successfully changed to model '{new_model_size}'")
            return True
        else:
            print(f"❌ Failed to change model, reverting to '{old_model_size}'")
            self.model_size = old_model_size
            self.load_model()
            return False

    def set_language(self, language: Optional[str]):
        """Set the target language for transcription"""
        self.language = language
        print(f"🌍 Language set to: {language or 'auto-detect'}")


class IntegratedSpeechSystem:
    """Integrated system combining Audio Engine and Speech Recognition"""

    def __init__(
        self,
        device_manager: AudioDeviceManager,
        whisper_model: str = "base",
        whisper_device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.device_manager = device_manager
        self.audio_engine = AudioEngine(device_manager)
        self.speech_engine = SpeechRecognitionEngine(
            whisper_model, whisper_device, language
        )

        # Connect the systems
        self._setup_integration()

        print("🤝 Integrated Speech System initialized")

    def _setup_integration(self):
        """Set up integration between audio and speech recognition"""

        def on_speech_detected(audio_data: np.ndarray):
            """Handle speech detection from audio engine"""
            print("🎤 Speech detected, starting transcription...")

            # Queue for async transcription
            self.speech_engine.queue_for_transcription(
                audio_data, self.audio_engine.sample_rate
            )

        def on_transcription_complete(text: str, result: Dict):
            """Handle completed transcription"""
            print(f'✅ Transcription: "{text}"')

            # This is where we'll later add AI response generation
            if hasattr(self, "on_user_speech"):
                self.on_user_speech(text, result)

        # Set up callbacks
        self.audio_engine.set_callbacks(on_speech_end=on_speech_detected)

        self.speech_engine.set_callbacks(
            on_transcription_complete=on_transcription_complete
        )

    def initialize(self) -> bool:
        """Initialize both engines"""
        print("🚀 Initializing integrated speech system...")

        # Load speech recognition model
        if not self.speech_engine.load_model():
            return False

        # Start async processing
        self.speech_engine.start_async_processing()

        print("✅ Integrated speech system ready!")
        return True

    def start_listening(self):
        """Start the complete speech-to-text pipeline"""
        print("👂 Starting speech-to-text pipeline...")
        return self.audio_engine.start_listening()

    def stop_listening(self):
        """Stop the speech-to-text pipeline"""
        print("🔇 Stopping speech-to-text pipeline...")
        self.audio_engine.stop_listening()
        self.speech_engine.stop_async_processing()

    def set_user_speech_callback(self, callback: Callable[[str, Dict], None]):
        """Set callback for when user speech is transcribed"""
        self.on_user_speech = callback


def main():
    """Demo/test function for speech recognition"""
    print("🎤 Speech Recognition Engine Test")
    print("=" * 50)

    # Initialize device manager
    device_manager = AudioDeviceManager()

    # Check if devices are configured
    if device_manager.selected_input_device is None:
        print("⚠️ Audio devices not configured. Running setup...")
        device_manager.interactive_setup()

    if device_manager.selected_input_device is None:
        print("❌ Cannot proceed without input device configured")
        return

    # Test menu
    while True:
        print(f"\n🎤 Speech Recognition Test Menu")
        print(f"[1] Test individual speech recognition engine")
        print(f"[2] Test integrated speech system")
        print(f"[3] Transcribe audio file")
        print(f"[4] Change Whisper model")
        print(f"[q] Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "1":
            # Test individual speech recognition
            model_size = (
                input(
                    "Whisper model (tiny/base/small/medium/large, default base): "
                ).strip()
                or "base"
            )

            speech_engine = SpeechRecognitionEngine(model_size)

            if speech_engine.load_model():
                print(
                    "🎤 Speak into your microphone, then press Enter to transcribe..."
                )

                # Record audio manually
                audio_engine = AudioEngine(device_manager)

                def on_speech_end(audio_data):
                    print("🔄 Transcribing...")
                    try:
                        result = speech_engine.transcribe_audio(audio_data)
                        print(f"📊 Stats: {speech_engine.get_statistics()}")
                    except Exception as e:
                        print(f"❌ Transcription error: {e}")

                audio_engine.set_callbacks(on_speech_end=on_speech_end)

                try:
                    audio_engine.start_listening()
                    input("Press Enter when done speaking...")
                    audio_engine.stop_listening()
                except KeyboardInterrupt:
                    audio_engine.stop_listening()

        elif choice == "2":
            # Test integrated system
            model_size = (
                input(
                    "Whisper model (tiny/base/small/medium/large, default base): "
                ).strip()
                or "base"
            )

            integrated_system = IntegratedSpeechSystem(device_manager, model_size)

            def on_user_speech(text: str, result: Dict):
                print(f'🎯 USER SAID: "{text}"')
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")

            integrated_system.set_user_speech_callback(on_user_speech)

            if integrated_system.initialize():
                try:
                    print("🎤 Integrated system listening... Speak normally!")
                    print("Press Ctrl+C to stop")

                    integrated_system.start_listening()

                    while True:
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    integrated_system.stop_listening()
                    print("\n🔇 Integrated system stopped")

        elif choice == "3":
            # Transcribe file
            filename = input("Audio file path: ").strip()
            model_size = input("Whisper model (default base): ").strip() or "base"

            speech_engine = SpeechRecognitionEngine(model_size)

            if speech_engine.load_model():
                try:
                    result = speech_engine.transcribe_file(filename)
                    print(f"\n📝 Transcription Results:")
                    print(f"Text: {result['text']}")
                    print(f"Language: {result['language']}")
                    print(f"Segments: {len(result['segments'])}")
                except Exception as e:
                    print(f"❌ File transcription error: {e}")

        elif choice == "4":
            # Change model
            print("Available models: tiny, base, small, medium, large")
            new_model = input("Enter new model: ").strip()

            if new_model in ["tiny", "base", "small", "medium", "large"]:
                speech_engine = SpeechRecognitionEngine(new_model)
                speech_engine.load_model()
            else:
                print("❌ Invalid model name")

        elif choice == "q":
            break

        else:
            print("❌ Invalid choice")

    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
