import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pyaudio
import json
import os
import numpy as np
import threading
import sys
import importlib.util

class AudioDeviceSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Voice Assistant - Device Setup & Transcription")
        self.root.geometry("600x600")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Variables to store selected devices
        self.selected_input = tk.StringVar()
        self.selected_output = tk.StringVar()
        
        # Storage for device info
        self.input_devices = {}
        self.output_devices = {}
        
        # Transcription variables
        self.transcriber = None
        self.is_transcribing = False
        
        # Check if transcription module is available
        self.transcription_available = self._check_transcription_module()
        
        self.setup_ui()
        self.populate_devices()
        self.load_settings()
    
    def _check_transcription_module(self):
        """Check if the transcription module and its dependencies are available"""
        missing_packages = []
        error_details = []
        
        # Check speech_recognition (essential)
        try:
            spec = importlib.util.find_spec("speech_recognition")
            if spec is None:
                missing_packages.append("speech_recognition")
            else:
                try:
                    import speech_recognition as sr
                    # Test basic functionality
                    recognizer = sr.Recognizer()
                except Exception as e:
                    missing_packages.append("speech_recognition")
                    error_details.append(f"speech_recognition import error: {str(e)}")
        except Exception as e:
            missing_packages.append("speech_recognition") 
            error_details.append(f"speech_recognition check error: {str(e)}")
        
        # Check pocketsphinx (optional - only for offline transcription)
        pocketsphinx_available = True
        try:
            spec = importlib.util.find_spec("pocketsphinx")
            if spec is None:
                pocketsphinx_available = False
                error_details.append("pocketsphinx not available (offline transcription disabled)")
            else:
                try:
                    import pocketsphinx
                except Exception as e:
                    pocketsphinx_available = False
                    error_details.append(f"pocketsphinx import error: {str(e)} (offline transcription disabled)")
        except Exception as e:
            pocketsphinx_available = False
            error_details.append(f"pocketsphinx check error: {str(e)} (offline transcription disabled)")
        
        # Check transcription module
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from processing.transcription import AudioTranscriber
        except ImportError as e:
            missing_packages.append("transcription_module")
            error_details.append(f"Transcription module import error: {str(e)}")
            return False
        except Exception as e:
            error_details.append(f"Transcription module error: {str(e)}")
            return False
        
        # Store info
        self.transcription_errors = error_details
        self.missing_packages = missing_packages
        self.pocketsphinx_available = pocketsphinx_available
        
        # Return True if at least speech_recognition works (Google transcription)
        return "speech_recognition" not in missing_packages
    
    def setup_ui(self):
        # Main frame with scrollable content
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Voice Assistant", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Device Setup & Speech Transcription", 
                                  font=("Arial", 12))
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Device Configuration Section
        device_frame = ttk.LabelFrame(main_frame, text="Audio Device Configuration", padding="10")
        device_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Input device section
        input_label = ttk.Label(device_frame, text="Input Device (Microphone):", 
                               font=("Arial", 10, "bold"))
        input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.input_combo = ttk.Combobox(device_frame, textvariable=self.selected_input,
                                       state="readonly", width=60)
        self.input_combo.grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=(tk.W, tk.E))
        
        # Output device section
        output_label = ttk.Label(device_frame, text="Output Device (Speakers):", 
                                font=("Arial", 10, "bold"))
        output_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.output_combo = ttk.Combobox(device_frame, textvariable=self.selected_output,
                                        state="readonly", width=60)
        self.output_combo.grid(row=3, column=0, columnspan=2, pady=(0, 15), sticky=(tk.W, tk.E))
        
        # Device test buttons
        device_button_frame = ttk.Frame(device_frame)
        device_button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        test_input_button = ttk.Button(device_button_frame, text="Test Input", 
                                      command=self.test_input)
        test_input_button.pack(side=tk.LEFT, padx=(0, 10))
        
        test_output_button = ttk.Button(device_button_frame, text="Test Output", 
                                       command=self.test_output)
        test_output_button.pack(side=tk.LEFT, padx=(0, 10))
        
        refresh_button = ttk.Button(device_button_frame, text="Refresh Devices", 
                                   command=self.refresh_devices)
        refresh_button.pack(side=tk.LEFT, padx=(0, 10))
        
        save_button = ttk.Button(device_button_frame, text="Save Settings", 
                                command=self.save_settings)
        save_button.pack(side=tk.LEFT)
        
        # Transcription Section
        transcription_frame = ttk.LabelFrame(main_frame, text="Speech Transcription", padding="10")
        transcription_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        if not self.transcription_available:
            # Show installation instructions if transcription is not available
            warning_label = ttk.Label(transcription_frame, 
                                    text="⚠️ Transcription module not available", 
                                    foreground="orange", font=("Arial", 10, "bold"))
            warning_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
            
            # Show detailed error information
            if hasattr(self, 'missing_packages') and self.missing_packages:
                missing_text = f"Missing packages: {', '.join(self.missing_packages)}"
                missing_label = ttk.Label(transcription_frame, text=missing_text, foreground="red")
                missing_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))
            
            if hasattr(self, 'transcription_errors') and self.transcription_errors:
                error_text = "\n".join(self.transcription_errors[:3])  # Show first 3 errors
                error_label = ttk.Label(transcription_frame, text=error_text, 
                                      foreground="gray", wraplength=500, justify="left")
                error_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))
            
            install_text = ttk.Label(transcription_frame, 
                                   text="Try these installation commands:",
                                   font=("Arial", 9, "bold"))
            install_text.grid(row=3, column=0, columnspan=2, pady=(0, 5))
            
            commands_text = ttk.Label(transcription_frame, 
                                   text="pip install --upgrade SpeechRecognition\npip install --upgrade pocketsphinx\npip install --upgrade pyaudio",
                                   foreground="blue", font=("Courier", 8))
            commands_text.grid(row=4, column=0, columnspan=2, pady=(0, 10))
            
            # Add buttons to help with debugging
            debug_frame = ttk.Frame(transcription_frame)
            debug_frame.grid(row=5, column=0, columnspan=2, pady=(0, 10))
            
            check_button = ttk.Button(debug_frame, text="Recheck Packages", 
                                    command=self.recheck_transcription)
            check_button.pack(side=tk.LEFT, padx=(0, 10))
            
            debug_button = ttk.Button(debug_frame, text="Show Debug Info", 
                                    command=self.show_debug_info)
            debug_button.pack(side=tk.LEFT)
        else:
            # Transcription controls
            trans_control_frame = ttk.Frame(transcription_frame)
            trans_control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Engine selection - adjust based on available packages
            ttk.Label(trans_control_frame, text="Engine:").pack(side=tk.LEFT, padx=(0, 5))
            self.engine_var = tk.StringVar(value="google")
            
            # Set available engines based on what's installed
            available_engines = ["google"]
            if hasattr(self, 'pocketsphinx_available') and self.pocketsphinx_available:
                available_engines.append("sphinx")
            
            engine_combo = ttk.Combobox(trans_control_frame, textvariable=self.engine_var,
                                       values=available_engines, state="readonly", width=15)
            engine_combo.pack(side=tk.LEFT, padx=(0, 20))
            
            # Show warning if sphinx is not available
            if not hasattr(self, 'pocketsphinx_available') or not self.pocketsphinx_available:
                sphinx_warning = ttk.Label(trans_control_frame, text="(offline unavailable)", 
                                         foreground="orange", font=("Arial", 8))
                sphinx_warning.pack(side=tk.LEFT, padx=(0, 20))
            
            # Recording buttons
            self.start_record_button = ttk.Button(trans_control_frame, text="Start Recording", 
                                                 command=self.start_recording, state="normal")
            self.start_record_button.pack(side=tk.LEFT, padx=(0, 10))
            
            self.stop_record_button = ttk.Button(trans_control_frame, text="Stop & Transcribe", 
                                                command=self.stop_and_transcribe, state="disabled")
            self.stop_record_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # Quick record button
            quick_record_button = ttk.Button(trans_control_frame, text="Quick Record (5s)", 
                                           command=self.quick_record)
            quick_record_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # Test audio button (new)
            test_audio_button = ttk.Button(trans_control_frame, text="Test Audio (3s)", 
                                         command=self.test_audio_capture)
            test_audio_button.pack(side=tk.LEFT)
            
            # Recording status
            self.recording_status = ttk.Label(transcription_frame, text="Ready to record", 
                                            foreground="green")
            self.recording_status.grid(row=1, column=0, columnspan=2, pady=(0, 10))
            
            # Transcription results
            ttk.Label(transcription_frame, text="Transcription Results:", 
                     font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
            
            self.transcription_text = scrolledtext.ScrolledText(transcription_frame, 
                                                               height=8, width=70, wrap=tk.WORD)
            self.transcription_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Button frame for clear and copy error buttons
            button_frame = ttk.Frame(transcription_frame)
            button_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
            
            # Clear results button
            clear_button = ttk.Button(button_frame, text="Clear Results", 
                                    command=self.clear_transcription)
            clear_button.pack(side=tk.LEFT)
            
            # Copy error button (initially hidden)
            self.copy_error_button = ttk.Button(button_frame, text="Copy Error", 
                                              command=self.copy_error_to_clipboard,
                                              state="disabled")
            self.copy_error_button.pack(side=tk.LEFT, padx=(10, 0))
            
            # Store the last error for copying
            self.last_error = ""
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                     foreground="green")
        self.status_label.pack()
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main_frame.columnconfigure(0, weight=1)
        device_frame.columnconfigure(0, weight=1)
        if self.transcription_available:
            transcription_frame.columnconfigure(0, weight=1)
    
    def populate_devices(self):
        """Scan for available audio devices"""
        try:
            self.input_devices.clear()
            self.output_devices.clear()
            
            device_count = self.audio.get_device_count()
            
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                device_name = device_info['name']
                
                # Check if device supports input (recording)
                if device_info['maxInputChannels'] > 0:
                    display_name = f"{device_name} (Index: {i})"
                    self.input_devices[display_name] = {
                        'index': i,
                        'info': device_info
                    }
                
                # Check if device supports output (playback)
                if device_info['maxOutputChannels'] > 0:
                    display_name = f"{device_name} (Index: {i})"
                    self.output_devices[display_name] = {
                        'index': i,
                        'info': device_info
                    }
            
            # Update comboboxes
            self.input_combo['values'] = list(self.input_devices.keys())
            self.output_combo['values'] = list(self.output_devices.keys())
            
            # Set default selections if available
            if self.input_devices:
                self.input_combo.current(0)
            if self.output_devices:
                self.output_combo.current(0)
                
            self.update_status(f"Found {len(self.input_devices)} input and {len(self.output_devices)} output devices")
            
            # Initialize transcriber with selected device
            if self.transcription_available:
                self._initialize_transcriber()
            
        except Exception as e:
            self.update_status(f"Error scanning devices: {str(e)}", "error")
    
    def _initialize_transcriber(self):
        """Initialize the transcriber with the selected input device"""
        if not self.transcription_available:
            return
        
        try:
            from processing.transcription import AudioTranscriber
            
            # Get selected input device
            device_index = None
            if self.selected_input.get() and self.selected_input.get() in self.input_devices:
                device_index = self.input_devices[self.selected_input.get()]['index']
            
            # Clean up old transcriber if it exists
            if hasattr(self, 'transcriber') and self.transcriber:
                self.transcriber.cleanup()
            
            # Create transcriber
            self.transcriber = AudioTranscriber(device_index=device_index)
            
            # Set up callbacks
            self.transcriber.set_callbacks(
                on_transcription=self._on_transcription_complete,
                on_error=self._on_transcription_error,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop
            )
            
        except Exception as e:
            self.update_status(f"Error initializing transcriber: {str(e)}", "error")
    
    def _on_transcription_complete(self, text):
        """Callback when transcription is complete"""
        def update_ui():
            current_text = self.transcription_text.get(1.0, tk.END)
            timestamp = self._get_timestamp()
            new_text = f"[{timestamp}] {text}\n\n"
            self.transcription_text.insert(tk.END, new_text)
            self.transcription_text.see(tk.END)  # Scroll to bottom
            self.update_status("Transcription complete!", "success")
        
        # Update UI in main thread
        self.root.after(0, update_ui)
    
    def _on_transcription_error(self, error):
        """Callback when transcription error occurs"""
        def update_ui():
            self.update_status(f"Transcription error: {error}", "error")
            self.recording_status.config(text="Error occurred", foreground="red")
            
            # Store error and enable copy button
            self.last_error = f"Transcription Error: {error}\nDevice: {self.selected_input.get()}\nEngine: {getattr(self, 'engine_var', None) and self.engine_var.get()}\nTimestamp: {self._get_timestamp()}"
            self.copy_error_button.config(state="normal")
            
            # Also add error to transcription text area
            timestamp = self._get_timestamp()
            error_text = f"[{timestamp}] ❌ ERROR: {error}\n\n"
            self.transcription_text.insert(tk.END, error_text)
            self.transcription_text.see(tk.END)
        
        self.root.after(0, update_ui)
    
    def _on_recording_start(self):
        """Callback when recording starts"""
        def update_ui():
            self.recording_status.config(text="🔴 Recording...", foreground="red")
            self.start_record_button.config(state="disabled")
            self.stop_record_button.config(state="normal")
            self.is_transcribing = True
        
        self.root.after(0, update_ui)
    
    def _on_recording_stop(self):
        """Callback when recording stops"""
        def update_ui():
            self.recording_status.config(text="Processing...", foreground="orange")
            self.start_record_button.config(state="normal")
            self.stop_record_button.config(state="disabled")
            self.is_transcribing = False
        
        self.root.after(0, update_ui)
    
    def start_recording(self):
        """Start recording audio"""
        if not self.transcription_available or not self.transcriber:
            messagebox.showwarning("Not Available", "Transcription not available. Please install required packages.")
            return
        
        if not self.selected_input.get():
            messagebox.showwarning("No Device", "Please select an input device first")
            return
        
        # Reinitialize transcriber in case device changed
        self._initialize_transcriber()
        
        success = self.transcriber.start_recording()
        if not success:
            self.update_status("Failed to start recording", "error")
    
    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio"""
        if not self.transcriber or not self.is_transcribing:
            return
        
        self.transcriber.stop_recording()
        
        # Start transcription in a separate thread
        def transcribe():
            self.transcriber.transcribe_recorded_audio(engine=self.engine_var.get())
        
        threading.Thread(target=transcribe, daemon=True).start()
    
    def quick_record(self):
        """Record for 5 seconds and transcribe"""
        if not self.transcription_available or not self.transcriber:
            messagebox.showwarning("Not Available", "Transcription not available. Please install required packages.")
            return
        
        if not self.selected_input.get():
            messagebox.showwarning("No Device", "Please select an input device first")
            return
        
        # Reinitialize transcriber in case device changed
        self._initialize_transcriber()
        
        def record_and_transcribe():
            self.transcriber.record_and_transcribe(duration=5.0, engine=self.engine_var.get())
        
        threading.Thread(target=record_and_transcribe, daemon=True).start()
    
    def test_audio_capture(self):
        """Test audio capture without transcription to check audio levels"""
        if not self.transcription_available or not self.transcriber:
            messagebox.showwarning("Not Available", "Transcription not available. Please install required packages.")
            return
        
        if not self.selected_input.get():
            messagebox.showwarning("No Device", "Please select an input device first")
            return
        
        # Reinitialize transcriber in case device changed
        self._initialize_transcriber()
        
        def test_audio():
            try:
                self.transcriber.start_recording()
                time.sleep(3.0)  # Record for 3 seconds
                audio_data = self.transcriber.stop_recording()
                
                if audio_data:
                    # Analyze the captured audio without transcribing
                    import tempfile
                    import wave
                    import numpy as np
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        
                        with wave.open(temp_filename, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(self.transcriber.audio.get_sample_size(self.transcriber.format))
                            wav_file.setframerate(self.transcriber.sample_rate)
                            wav_file.writeframes(b''.join(audio_data))
                        
                        # Analyze the audio
                        analysis = self.transcriber._analyze_audio_quality(temp_filename)
                        
                        # Show results in transcription area
                        def show_results():
                            timestamp = self._get_timestamp()
                            test_text = f"[{timestamp}] 🔍 AUDIO TEST: {analysis}\n\n"
                            self.transcription_text.insert(tk.END, test_text)
                            self.transcription_text.see(tk.END)
                            self.update_status("Audio test complete", "success")
                        
                        self.root.after(0, show_results)
                        
                        # Clean up
                        self.transcriber._safe_delete_file(temp_filename)
                        
            except Exception as e:
                def show_error():
                    self.update_status(f"Audio test failed: {str(e)}", "error")
                self.root.after(0, show_error)
        
        threading.Thread(target=test_audio, daemon=True).start()
    
    def recheck_transcription(self):
        """Recheck if transcription packages are available"""
        self.transcription_available = self._check_transcription_module()
        if self.transcription_available:
            messagebox.showinfo("Success", "Transcription packages found! Please restart the application.")
            self.update_status("Transcription packages found - restart app to use", "success")
        else:
            error_msg = "Still missing packages:\n"
            if hasattr(self, 'missing_packages'):
                error_msg += f"Missing: {', '.join(self.missing_packages)}\n"
            if hasattr(self, 'transcription_errors'):
                error_msg += f"Errors: {'; '.join(self.transcription_errors[:2])}"
            messagebox.showwarning("Still Missing", error_msg)
    
    def show_debug_info(self):
        """Show detailed debug information"""
        import sys
        import platform
        
        debug_info = f"""Python Debug Information:
Python Version: {sys.version}
Platform: {platform.platform()}
Python Executable: {sys.executable}
Python Path: {sys.path[:3]}...

Package Check Results:
"""
        if hasattr(self, 'missing_packages'):
            debug_info += f"Missing Packages: {self.missing_packages}\n"
        if hasattr(self, 'transcription_errors'):
            debug_info += f"Import Errors:\n" + "\n".join(self.transcription_errors)
        
        # Try to get more specific package info
        try:
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            speech_installed = any('speech' in pkg.lower() for pkg in installed_packages)
            sphinx_installed = any('sphinx' in pkg.lower() for pkg in installed_packages)
            debug_info += f"\nInstalled packages with 'speech': {speech_installed}"
            debug_info += f"\nInstalled packages with 'sphinx': {sphinx_installed}"
        except:
            debug_info += "\nCould not check installed packages"
        
        # Show in a dialog
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Information")
        debug_window.geometry("600x400")
        
        text_widget = scrolledtext.ScrolledText(debug_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, debug_info)
        
        # Add copy button
        copy_button = ttk.Button(debug_window, text="Copy to Clipboard", 
                               command=lambda: self._copy_to_clipboard(debug_info))
        copy_button.pack(pady=5)
    
    def _copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copied", "Debug information copied to clipboard!")
    
    def clear_transcription(self):
        """Clear the transcription results"""
        self.transcription_text.delete(1.0, tk.END)
        # Reset error state
        self.last_error = ""
        self.copy_error_button.config(state="disabled")
    
    def copy_error_to_clipboard(self):
        """Copy the last error to clipboard"""
        if self.last_error:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_error)
            self.root.update()  # Ensure clipboard is updated
            self.update_status("Error copied to clipboard!", "success")
            messagebox.showinfo("Copied", "Error details copied to clipboard!")
        else:
            messagebox.showwarning("No Error", "No error to copy")
    
    def _get_timestamp(self):
        """Get current timestamp for transcription entries"""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    def test_input(self):
        """Basic test to see if input device is working"""
        if not self.selected_input.get():
            messagebox.showwarning("No Device", "Please select an input device first")
            return
        
        try:
            device_name = self.selected_input.get()
            device_index = self.input_devices[device_name]['index']
            device_info = self.input_devices[device_name]['info']
            
            # Use device's default sample rate if available
            sample_rate = int(device_info.get('defaultSampleRate', 44100))
            
            # Try to open a stream briefly to test the device
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            
            # Read a small amount of data
            stream.read(1024)
            stream.stop_stream()
            stream.close()
            
            self.update_status("Input device test successful!", "success")
            messagebox.showinfo("Test Result", f"Input device '{device_name}' is accessible and working!")
            
        except Exception as e:
            self.update_status(f"Input test failed: {str(e)}", "error")
            messagebox.showerror("Test Failed", f"Could not access input device:\n{str(e)}")
    
    def test_output(self):
        """Test output device by playing a simple tone"""
        if not self.selected_output.get():
            messagebox.showwarning("No Device", "Please select an output device first")
            return
        
        try:
            device_name = self.selected_output.get()
            device_index = self.output_devices[device_name]['index']
            device_info = self.output_devices[device_name]['info']
            
            # Use device's default sample rate if available
            sample_rate = int(device_info.get('defaultSampleRate', 44100))
            
            # Generate a simple sine wave tone
            duration = 1.0  # seconds
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.3  # Lower volume
            
            # Convert to 16-bit integers
            audio_data = (wave * 32767).astype(np.int16)
            
            # Open output stream
            output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=device_index
            )
            
            # Play the tone
            output_stream.write(audio_data.tobytes())
            output_stream.stop_stream()
            output_stream.close()
            
            self.update_status("Output test completed!", "success")
            messagebox.showinfo("Test Result", f"Test tone played on '{device_name}'")
            
        except Exception as e:
            self.update_status(f"Output test failed: {str(e)}", "error")
            messagebox.showerror("Test Failed", f"Could not test output device:\n{str(e)}")
    
    def refresh_devices(self):
        """Refresh the device list"""
        self.populate_devices()
        self.update_status("Device list refreshed")
    
    def save_settings(self):
        """Save device selections to a JSON file"""
        if not self.selected_input.get() or not self.selected_output.get():
            messagebox.showwarning("Incomplete Selection", 
                                 "Please select both input and output devices")
            return
        
        settings = {
            'input_device': {
                'name': self.selected_input.get(),
                'index': self.input_devices[self.selected_input.get()]['index']
            },
            'output_device': {
                'name': self.selected_output.get(),
                'index': self.output_devices[self.selected_output.get()]['index']
            }
        }
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            with open('data/audio_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.update_status("Settings saved successfully!", "success")
            messagebox.showinfo("Saved", "Audio device settings have been saved!")
            
        except Exception as e:
            self.update_status(f"Error saving settings: {str(e)}", "error")
            messagebox.showerror("Save Error", f"Could not save settings:\n{str(e)}")
    
    def load_settings(self):
        """Load previously saved device selections"""
        try:
            if os.path.exists('data/audio_settings.json'):
                with open('data/audio_settings.json', 'r') as f:
                    settings = json.load(f)
                
                # Try to restore selections
                input_name = settings.get('input_device', {}).get('name', '')
                output_name = settings.get('output_device', {}).get('name', '')
                
                if input_name in self.input_devices:
                    self.selected_input.set(input_name)
                
                if output_name in self.output_devices:
                    self.selected_output.set(output_name)
                
                self.update_status("Previous settings loaded")
                
        except Exception as e:
            self.update_status(f"Could not load previous settings: {str(e)}", "warning")
    
    def update_status(self, message, status_type="info"):
        """Update the status label"""
        colors = {
            "info": "black",
            "success": "green", 
            "warning": "orange",
            "error": "red"
        }
        
        self.status_label.config(text=message, foreground=colors.get(status_type, "black"))
    
    def get_selected_devices(self):
        """Return the currently selected devices"""
        input_device = None
        output_device = None
        
        if self.selected_input.get():
            input_device = self.input_devices[self.selected_input.get()]
        
        if self.selected_output.get():
            output_device = self.output_devices[self.selected_output.get()]
        
        return input_device, output_device
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'transcriber') and self.transcriber:
            self.transcriber.cleanup()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def __del__(self):
        """Clean up PyAudio"""
        self.cleanup()


def main():
    root = tk.Tk()
    app = AudioDeviceSelector(root)
    
    # Handle window closing
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()