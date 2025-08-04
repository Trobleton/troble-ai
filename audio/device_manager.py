import tkinter as tk
from tkinter import ttk, messagebox
import pyaudio
import json
import os
import numpy as np

class AudioDeviceSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Device Selector")
        self.root.geometry("500x400")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Variables to store selected devices
        self.selected_input = tk.StringVar()
        self.selected_output = tk.StringVar()
        
        # Storage for device info
        self.input_devices = {}
        self.output_devices = {}
        
        self.setup_ui()
        self.populate_devices()
        self.load_settings()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Audio Device Configuration", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input device section
        input_label = ttk.Label(main_frame, text="Input Device (Microphone):", 
                               font=("Arial", 12, "bold"))
        input_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.input_combo = ttk.Combobox(main_frame, textvariable=self.selected_input,
                                       state="readonly", width=50)
        self.input_combo.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Output device section
        output_label = ttk.Label(main_frame, text="Output Device (Speakers):", 
                                font=("Arial", 12, "bold"))
        output_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        self.output_combo = ttk.Combobox(main_frame, textvariable=self.selected_output,
                                        state="readonly", width=50)
        self.output_combo.grid(row=4, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        # Test input button
        test_input_button = ttk.Button(button_frame, text="Test Input", 
                                      command=self.test_input)
        test_input_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Test output button
        test_output_button = ttk.Button(button_frame, text="Test Output", 
                                       command=self.test_output)
        test_output_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        refresh_button = ttk.Button(button_frame, text="Refresh Devices", 
                                   command=self.refresh_devices)
        refresh_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save button
        save_button = ttk.Button(button_frame, text="Save Settings", 
                                command=self.save_settings)
        save_button.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                     foreground="green")
        self.status_label.pack()
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
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
            
        except Exception as e:
            self.update_status(f"Error scanning devices: {str(e)}", "error")
    
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