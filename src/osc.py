from pythonosc import udp_client
import time

class VRChatOSC:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_message(self, text, visible=True):
        self.client.send_message("/chatbox/input", [text, visible, False])

    def send_notification(self, text, duration=3.0):
        # Send notification-style text
        self.client.send_message("/chatbox/input", [text, True, True])
        # Auto-clear after duration
        time.sleep(duration)
        self.clear_message()

    def clear_message(self):
        # Clear the chatbox message
        self.client.send_message("/chatbox/input", ["", True, False])

    def pending_message(self):
        # Show typing indicator
        self.client.send_message("/chatbox/typing", True)