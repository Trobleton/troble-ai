from pythonosc import udp_client, dispatcher, osc_server
import time
import threading

class VRChatOSC:
    def __init__(self, ip="127.0.0.1", send_port=9000, receive_port=9001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        
        # OSC server for receiving events
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer((ip, receive_port), self.dispatcher)
        self.server_thread = None
        self.callbacks = {}

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
        
    def send_avatar_parameter(self, param_name, param_value):
        self.client.send_message(f"/avatar/parameters/{param_name}", param_value)
    
    def request_avatar_info(self):
        """Request current avatar information from VRChat"""
        # VRChat responds to these requests with current avatar data
        self.client.send_message("/avatar/change", "")
        
    def request_all_parameters(self):
        """Request all current parameter values"""
        self.client.send_message("/avatar/parameters/request", "")
    
    def register_callback(self, address, callback):
        """Register a callback for a specific OSC address"""
        self.callbacks[address] = callback
        
        def handler(address_path, *args):
            callback(*args)
            
        self.dispatcher.map(address, handler)
    
    def start_server(self):
        """Start the OSC server in a separate thread"""
        if self.server_thread is None or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"OSC server started on port {self.server.server_address[1]}")
    
    def stop_server(self):
        """Stop the OSC server"""
        if self.server_thread and self.server_thread.is_alive():
            self.server.shutdown()
            self.server_thread.join()
            print("OSC server stopped")