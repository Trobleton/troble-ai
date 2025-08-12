from pythonosc import udp_client
import time

class VRChatOSC:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.is_listening = False

    def send_message(self, text, visible=True):
        # Filter out problematic Unicode characters for VRChat
        filtered_text = text.encode('ascii', errors='ignore').decode('ascii')
        
        # Prevent duplicate "listening..." messages
        if filtered_text.lower().startswith("listening") and self.is_listening:
            return
        
        # Update listening state only for listening messages
        if filtered_text.lower().startswith("listening"):
            self.is_listening = True
        elif filtered_text.lower().startswith("thinking") or filtered_text.strip() == "":
            # Don't change listening state for thinking messages or empty messages
            pass
        else:
            # Any other message (actual response) resets listening state
            # This allows "listening..." to show again for the next conversation turn
            self.is_listening = False
            
        self.client.send_message("/chatbox/input", [filtered_text, visible, False])

    def send_notification(self, text, duration=3.0):
        # Send notification-style text
        self.client.send_message("/chatbox/input", [text, True, True])
        # Auto-clear after duration
        time.sleep(duration)
        self.clear_message()

    def clear_message(self):
        # Clear the chatbox message
        self.is_listening = False
        self.client.send_message("/chatbox/input", ["", True, False])

    def pending_message(self):
        self.client.send_message("/chatbox/typing", True)