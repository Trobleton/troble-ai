from audio.device_manager import AudioDeviceSelector
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    device_selector = AudioDeviceSelector(root)
    root.mainloop()