import sounddevice as sd
import json
import os
from typing import Dict, List, Tuple, Optional


class AudioDeviceManager:
    def __init__(self, config_file: str = "audio_config.json"):
        self.config_file = config_file
        self.selected_input_device = None
        self.selected_output_device = None
        self.load_config()

    def get_available_devices(self) -> List[Dict]:
        """Get all available audio devices"""
        devices = sd.query_devices()
        device_list = []

        for i, device in enumerate(devices):
            device_info = {
                "index": i,
                "name": device["name"],
                "max_input_channels": device["max_input_channels"],
                "max_output_channels": device["max_output_channels"],
                "default_samplerate": device["default_samplerate"],
                "hostapi": sd.query_hostapis(device["hostapi"])["name"],
            }
            device_list.append(device_info)

        return device_list

    def display_devices(self):
        """Display all available audio devices in a formatted way"""
        devices = self.get_available_devices()

        print("\n" + "=" * 80)
        print("AVAILABLE AUDIO DEVICES")
        print("=" * 80)

        print("\nINPUT DEVICES (Microphones):")
        print("-" * 50)
        for device in devices:
            if device["max_input_channels"] > 0:
                status = (
                    " ✓ SELECTED"
                    if device["index"] == self.selected_input_device
                    else ""
                )
                print(
                    f"[{device['index']:2d}] {device['name']:<40} ({device['hostapi']}){status}"
                )

        print("\nOUTPUT DEVICES (Speakers/Headphones):")
        print("-" * 50)
        for device in devices:
            if device["max_output_channels"] > 0:
                status = (
                    " ✓ SELECTED"
                    if device["index"] == self.selected_output_device
                    else ""
                )
                print(
                    f"[{device['index']:2d}] {device['name']:<40} ({device['hostapi']}){status}"
                )

        print("\nDUAL DEVICES (Input + Output):")
        print("-" * 50)
        for device in devices:
            if device["max_input_channels"] > 0 and device["max_output_channels"] > 0:
                in_status = (
                    " ✓ INPUT" if device["index"] == self.selected_input_device else ""
                )
                out_status = (
                    " ✓ OUTPUT"
                    if device["index"] == self.selected_output_device
                    else ""
                )
                status = in_status + out_status
                print(
                    f"[{device['index']:2d}] {device['name']:<40} ({device['hostapi']}){status}"
                )

    def select_input_device(self, device_index: int) -> bool:
        """Select an input device by index"""
        devices = self.get_available_devices()

        if 0 <= device_index < len(devices):
            device = devices[device_index]
            if device["max_input_channels"] > 0:
                self.selected_input_device = device_index
                print(f"✓ Selected input device: {device['name']}")
                return True
            else:
                print(f"✗ Device '{device['name']}' has no input channels")
                return False
        else:
            print(f"✗ Invalid device index: {device_index}")
            return False

    def select_output_device(self, device_index: int) -> bool:
        """Select an output device by index"""
        devices = self.get_available_devices()

        if 0 <= device_index < len(devices):
            device = devices[device_index]
            if device["max_output_channels"] > 0:
                self.selected_output_device = device_index
                print(f"✓ Selected output device: {device['name']}")
                return True
            else:
                print(f"✗ Device '{device['name']}' has no output channels")
                return False
        else:
            print(f"✗ Invalid device index: {device_index}")
            return False

    def test_devices(self):
        """Test the selected audio devices"""
        if self.selected_input_device is None:
            print("✗ No input device selected")
            return False

        if self.selected_output_device is None:
            print("✗ No output device selected")
            return False

        try:
            # Test input device
            print(f"Testing input device...")
            with sd.InputStream(
                device=self.selected_input_device, channels=1, samplerate=44100
            ):
                print("✓ Input device is working")

            # Test output device
            print(f"Testing output device...")
            with sd.OutputStream(
                device=self.selected_output_device, channels=2, samplerate=44100
            ):
                print("✓ Output device is working")

            print("✓ Both devices are working correctly!")
            return True

        except Exception as e:
            print(f"✗ Device test failed: {e}")
            return False

    def save_config(self):
        """Save the current device selection to config file"""
        config = {
            "input_device": self.selected_input_device,
            "output_device": self.selected_output_device,
        }

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"✓ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"✗ Failed to save configuration: {e}")

    def load_config(self):
        """Load device selection from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)

                self.selected_input_device = config.get("input_device")
                self.selected_output_device = config.get("output_device")
                print(f"✓ Configuration loaded from {self.config_file}")

            except Exception as e:
                print(f"✗ Failed to load configuration: {e}")
        else:
            print(
                f"No configuration file found. Will create {self.config_file} when you save."
            )

    def get_device_info(self, device_index: int) -> Optional[Dict]:
        """Get detailed information about a specific device"""
        devices = self.get_available_devices()
        if 0 <= device_index < len(devices):
            return devices[device_index]
        return None

    def interactive_setup(self):
        """Interactive setup wizard for selecting devices"""
        print("\n" + "=" * 60)
        print("SPEECH-TO-SPEECH AI - AUDIO DEVICE SETUP")
        print("=" * 60)

        while True:
            self.display_devices()

            print(f"\nCurrent Selection:")
            print(
                f"Input Device:  {self.selected_input_device if self.selected_input_device is not None else 'None'}"
            )
            print(
                f"Output Device: {self.selected_output_device if self.selected_output_device is not None else 'None'}"
            )

            print(f"\nOptions:")
            print(f"[i] Select input device")
            print(f"[o] Select output device")
            print(f"[t] Test selected devices")
            print(f"[s] Save configuration")
            print(f"[q] Quit setup")

            choice = input("\nEnter your choice: ").lower().strip()

            if choice == "i":
                try:
                    device_index = int(input("Enter input device index: "))
                    self.select_input_device(device_index)
                except ValueError:
                    print("✗ Please enter a valid number")

            elif choice == "o":
                try:
                    device_index = int(input("Enter output device index: "))
                    self.select_output_device(device_index)
                except ValueError:
                    print("✗ Please enter a valid number")

            elif choice == "t":
                self.test_devices()

            elif choice == "s":
                self.save_config()

            elif choice == "q":
                break

            else:
                print("✗ Invalid choice. Please try again.")

            input("\nPress Enter to continue...")


def main():
    """Main function to run the audio device manager"""
    try:
        manager = AudioDeviceManager()
        manager.interactive_setup()

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
