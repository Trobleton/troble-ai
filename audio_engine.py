import sounddevice as sd
import numpy as np
import queue
import time
import wave
from typing import Optional, Callable, List
from audio_device_manager import AudioDeviceManager


class AudioEngine:
    def __init__(self, device_manager: AudioDeviceManager):
        self.device_manager = device_manager
        self.sample_rate = 16000  # Common rate for speech recognition
        self.channels = 1  # Mono for speech
        self.chunk_size = 1024  # Buffer size

        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None

        # Playback state
        self.is_playing = False
        self.playback_queue = queue.Queue()
        self.playback_thread = None

        # Voice Activity Detection parameters
        self.vad_threshold = 0.01  # Minimum volume to consider as speech
        self.silence_duration = 1.5  # Seconds of silence before stopping recording
        self.min_recording_duration = 0.5  # Minimum recording length

        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_audio_chunk = None

        # Audio buffers
        self.current_recording = []
        self.silence_start_time = None
        self.recording_start_time = None

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable[[np.ndarray], None]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """Set callback functions for audio events"""
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_audio_chunk = on_audio_chunk

    def calculate_volume(self, audio_data: np.ndarray) -> float:
        """Calculate RMS volume of audio data"""
        return np.sqrt(np.mean(audio_data**2))

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback function for audio input stream"""
        if status:
            print(f"Audio input status: {status}")

        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_chunk = np.mean(indata, axis=1)
        else:
            audio_chunk = indata.flatten()

        # Calculate volume for VAD
        volume = self.calculate_volume(audio_chunk)
        current_time = time.time()

        # Voice Activity Detection
        if volume > self.vad_threshold:
            # Speech detected
            if not self.is_recording:
                # Start recording
                self.is_recording = True
                self.recording_start_time = current_time
                self.current_recording = []
                self.silence_start_time = None

                if self.on_speech_start:
                    self.on_speech_start()

                print("🎤 Speech detected - Recording started")

            # Reset silence timer
            self.silence_start_time = None

        else:
            # Silence detected
            if self.is_recording:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                elif current_time - self.silence_start_time > self.silence_duration:
                    # Enough silence, stop recording
                    recording_duration = current_time - self.recording_start_time

                    if recording_duration >= self.min_recording_duration:
                        # Valid recording
                        complete_audio = np.concatenate(self.current_recording)

                        if self.on_speech_end:
                            self.on_speech_end(complete_audio)

                        print(
                            f"🔇 Recording stopped - Duration: {recording_duration:.2f}s"
                        )
                    else:
                        print(
                            f"🔇 Recording too short ({recording_duration:.2f}s) - Discarded"
                        )

                    self.is_recording = False
                    self.current_recording = []
                    self.silence_start_time = None

        # Store audio chunk if recording
        if self.is_recording:
            self.current_recording.append(audio_chunk.copy())

            if self.on_audio_chunk:
                self.on_audio_chunk(audio_chunk)

        # Add to queue for potential processing
        self.audio_queue.put((audio_chunk.copy(), volume, self.is_recording))

    def start_listening(self):
        """Start continuous audio listening with VAD"""
        if self.device_manager.selected_input_device is None:
            raise ValueError("No input device selected")

        print(
            f"🎧 Starting audio listener on device {self.device_manager.selected_input_device}"
        )
        print(f"📊 Sample rate: {self.sample_rate}Hz, Channels: {self.channels}")
        print(
            f"🔊 VAD threshold: {self.vad_threshold}, Silence duration: {self.silence_duration}s"
        )
        print("Listening for speech...")

        try:
            self.input_stream = sd.InputStream(
                device=self.device_manager.selected_input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                dtype=np.float32,
            )
            self.input_stream.start()
            return True

        except Exception as e:
            print(f"❌ Failed to start audio listener: {e}")
            return False

    def stop_listening(self):
        """Stop audio listening"""
        if hasattr(self, "input_stream"):
            self.input_stream.stop()
            self.input_stream.close()
            print("🔇 Audio listener stopped")

    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """Play audio data through selected output device"""
        if self.device_manager.selected_output_device is None:
            raise ValueError("No output device selected")

        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure audio is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure audio is in range [-1, 1]
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert mono to stereo if needed
        if len(audio_data.shape) == 1:
            audio_data = np.column_stack([audio_data, audio_data])

        print(f"🔊 Playing audio: {len(audio_data) / sample_rate:.2f}s duration")

        try:
            sd.play(
                audio_data,
                samplerate=sample_rate,
                device=self.device_manager.selected_output_device,
            )
            sd.wait()  # Wait for playback to complete
            print("✅ Audio playback completed")

        except Exception as e:
            print(f"❌ Audio playback failed: {e}")

    def save_audio_to_wav(
        self, audio_data: np.ndarray, filename: str, sample_rate: Optional[int] = None
    ):
        """Save audio data to WAV file"""
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"💾 Audio saved to {filename}")

    def load_audio_from_wav(self, filename: str) -> tuple[np.ndarray, int]:
        """Load audio data from WAV file"""
        with wave.open(filename, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0

        print(f"📁 Audio loaded from {filename}: {len(audio_data) / sample_rate:.2f}s")
        return audio_data, sample_rate

    def generate_test_tone(
        self, frequency: float = 440.0, duration: float = 1.0
    ) -> np.ndarray:
        """Generate a test tone for audio testing"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% volume
        return tone.astype(np.float32)

    def get_audio_levels(self, duration: float = 5.0) -> List[float]:
        """Monitor audio input levels for a specified duration"""
        if self.device_manager.selected_input_device is None:
            raise ValueError("No input device selected")

        levels = []
        print(f"📊 Monitoring audio levels for {duration} seconds...")

        def level_callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")

            volume = self.calculate_volume(indata.flatten())
            levels.append(volume)

            # Visual level indicator
            bar_length = int(volume * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"\r🎤 Level: [{bar}] {volume:.4f}", end="", flush=True)

        try:
            with sd.InputStream(
                device=self.device_manager.selected_input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=level_callback,
                dtype=np.float32,
            ):
                sd.sleep(int(duration * 1000))

            print(f"\n📈 Average level: {np.mean(levels):.4f}")
            print(f"📈 Peak level: {np.max(levels):.4f}")
            return levels

        except Exception as e:
            print(f"❌ Level monitoring failed: {e}")
            return []

    def calibrate_vad_threshold(self, duration: float = 10.0):
        """Auto-calibrate VAD threshold based on ambient noise"""
        print(f"🎯 Calibrating VAD threshold...")
        print(
            f"Please remain quiet for {duration} seconds to measure background noise..."
        )

        levels = self.get_audio_levels(duration)

        if levels:
            avg_noise = np.mean(levels)
            peak_noise = np.max(levels)

            # Set threshold to 3x average noise level, with minimum threshold
            suggested_threshold = max(avg_noise * 3, 0.005)

            print(f"\n🎯 Calibration results:")
            print(f"   Average noise level: {avg_noise:.4f}")
            print(f"   Peak noise level: {peak_noise:.4f}")
            print(f"   Suggested VAD threshold: {suggested_threshold:.4f}")
            print(f"   Current VAD threshold: {self.vad_threshold:.4f}")

            response = input("\nApply suggested threshold? (y/n): ").lower().strip()
            if response == "y":
                self.vad_threshold = suggested_threshold
                print(f"✅ VAD threshold updated to {self.vad_threshold:.4f}")
            else:
                print("❌ Keeping current threshold")
        else:
            print("❌ Calibration failed - no audio data received")


def main():
    """Demo/test function for the audio engine"""
    print("🎵 Audio Engine Test")
    print("=" * 50)

    # Initialize device manager
    device_manager = AudioDeviceManager()

    # Check if devices are configured
    if (
        device_manager.selected_input_device is None
        or device_manager.selected_output_device is None
    ):
        print("⚠️ Audio devices not configured. Running setup...")
        device_manager.interactive_setup()

    if (
        device_manager.selected_input_device is None
        or device_manager.selected_output_device is None
    ):
        print("❌ Cannot proceed without audio devices configured")
        return

    # Initialize audio engine
    audio_engine = AudioEngine(device_manager)

    # Set up callbacks
    def on_speech_start():
        print("🗣️ Speech started!")

    def on_speech_end(audio_data):
        print(
            f"✅ Speech ended! Captured {len(audio_data) / audio_engine.sample_rate:.2f} seconds"
        )

        # Save the recording
        filename = f"recording_{int(time.time())}.wav"
        audio_engine.save_audio_to_wav(audio_data, filename)

        # Play it back
        print("🔊 Playing back your recording...")
        time.sleep(0.5)  # Brief pause
        audio_engine.play_audio(audio_data)

    audio_engine.set_callbacks(
        on_speech_start=on_speech_start, on_speech_end=on_speech_end
    )

    # Test menu
    while True:
        print(f"\n🎵 Audio Engine Test Menu")
        print(f"[1] Test audio levels")
        print(f"[2] Calibrate VAD threshold")
        print(f"[3] Play test tone")
        print(f"[4] Start speech detection (Ctrl+C to stop)")
        print(f"[5] Load and play WAV file")
        print(f"[q] Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "1":
            try:
                duration = float(
                    input("Monitor duration (seconds, default 5): ") or "5"
                )
                audio_engine.get_audio_levels(duration)
            except ValueError:
                print("❌ Invalid duration")

        elif choice == "2":
            try:
                duration = float(
                    input("Calibration duration (seconds, default 10): ") or "10"
                )
                audio_engine.calibrate_vad_threshold(duration)
            except ValueError:
                print("❌ Invalid duration")

        elif choice == "3":
            try:
                freq = float(input("Frequency (Hz, default 440): ") or "440")
                duration = float(input("Duration (seconds, default 1): ") or "1")
                tone = audio_engine.generate_test_tone(freq, duration)
                audio_engine.play_audio(tone)
            except ValueError:
                print("❌ Invalid parameters")

        elif choice == "4":
            try:
                print(
                    "🎤 Starting speech detection... Speak normally and pause when done."
                )
                print("Press Ctrl+C to stop listening.")
                audio_engine.start_listening()

                # Keep running until interrupted
                while True:
                    time.sleep(0.1)

            except KeyboardInterrupt:
                audio_engine.stop_listening()
                print("\n🔇 Speech detection stopped")

        elif choice == "5":
            filename = input("WAV filename: ").strip()
            try:
                audio_data, sample_rate = audio_engine.load_audio_from_wav(filename)
                audio_engine.play_audio(audio_data, sample_rate)
            except Exception as e:
                print(f"❌ Error loading/playing file: {e}")

        elif choice == "q":
            break

        else:
            print("❌ Invalid choice")

    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
