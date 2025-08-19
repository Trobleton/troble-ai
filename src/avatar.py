import os
import json
from src.osc import VRChatOSC

class VRCAvatar:
    def __init__(self):
        self.vrchat_user_id = self._get_most_recent_user_id()
        
        self.vrchat_osc = VRChatOSC()
        self.avatar_id = "avtr_92982fd4-aedd-4adf-a5c6-e513b506dcd7"
        self.avatar_name = None
        self.avatar_path = os.path.join(os.environ['USERPROFILE'], 'AppData', 'LocalLow', 'VRChat', 'VRChat', "OSC", self.vrchat_user_id, "Avatars")
        
        # Register OSC callbacks for avatar changes and parameters
        self.vrchat_osc.register_callback("/avatar/change", self.on_avatar_change)
        self.vrchat_osc.register_callback("/avatar/parameters/*", self.on_avatar_parameter)
        self.vrchat_osc.start_server()
        
        # Request current avatar info on startup
        self.get_current_avatar_name()
    
    def _get_most_recent_user_id(self):
        osc_path = os.path.join(os.environ['USERPROFILE'], 'AppData', 'LocalLow', 'VRChat', 'VRChat', "OSC")
        if not os.path.exists(osc_path):
            return None
            
        user_dirs = [d for d in os.listdir(osc_path) if os.path.isdir(os.path.join(osc_path, d))]
        if not user_dirs:
            return None
            
        full_paths = [os.path.join(osc_path, d) for d in user_dirs]
        most_recent_dir = max(full_paths, key=os.path.getmtime)
        return os.path.basename(most_recent_dir)
        
    def _get_most_recent_avatar_file(self):
        if not self.vrchat_user_id:
            return None
            
        if not os.path.exists(self.avatar_path):
            return None
            
        try:
            avatar_dir = os.listdir(self.avatar_path)
            full_paths = [os.path.join(self.avatar_path, f) for f in avatar_dir]
            json_files = [p for p in full_paths if os.path.isfile(p) and p.lower().endswith('.json')]
            
            if not json_files:
                return None
                
            most_recent_file = max(json_files, key=os.path.getmtime)
            return most_recent_file
            
        except Exception:
            return None

    def get_current_avatar(self):
        """Get current avatar data - tries to use current avatar ID first, falls back to most recent file"""
        if not self.vrchat_user_id:
            print("Error: Could not find VRChat user ID")
            return None
            
        if not os.path.exists(self.avatar_path):
            print(f"Error: Avatar path does not exist: {self.avatar_path}")
            return None
        
        avatar_json_path = None
        
        # First try to use the current avatar ID if we have it
        if self.avatar_id:
            potential_path = os.path.join(self.avatar_path, f"{self.avatar_id}.json")
            if os.path.exists(potential_path):
                avatar_json_path = potential_path
                print(f"Using current avatar ID: {self.avatar_id}")
        
        # Fall back to most recent file if current avatar file not found
        if not avatar_json_path:
            avatar_json_path = self._get_most_recent_avatar_file()
            if avatar_json_path:
                print("Using most recent avatar file as fallback")
        
        if not avatar_json_path:
            print("Error: No JSON files found in avatar directory")
            return None
            
        print(f"Reading avatar file: {avatar_json_path}")
        
        try:
            with open(avatar_json_path, 'r', encoding='utf-8-sig') as f:
                try:
                    d = json.load(f)
                    print(d["name"])
                    return d
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in file {avatar_json_path}: {e}")
                    print(f"Error at line {e.lineno}, column {e.colno}")
                    return None
                    
        except Exception as e:
            print(f"Error reading avatar file: {e}")
            return None
        
    def set_avatar_color(self, color):
        avatar_data = self.get_current_avatar()
        if not avatar_data:
            print("Error: Could not retrieve avatar data")
            return

        # Find the name of the avatar clothing color param
        params = avatar_data.get('parameters')
        if not isinstance(params, list):
            print("Error: avatar parameters missing or invalid")
            return

        clothing_param = next(
            (p for p in params
             if isinstance(p, dict) and 'name' in p and 'clothing/color' in p['name'].lower()),
            None
        )

        if not clothing_param:
            print('Error: No parameter found with name including "Clothing/Color"')
            return

        clothing_param_name = clothing_param['name']
        print(f'Found clothing color parameter: {clothing_param_name}')

        # Set the avatar color using the VRChatOSC instance
        self.vrchat_osc.send_avatar_parameter(clothing_param_name, color)
    
    def get_current_avatar_name(self):
        """Get the current avatar name"""
        if not self.avatar_name and self.avatar_id:
            # Try to get name from current avatar data
            avatar_data = self.get_current_avatar()
            if avatar_data and 'name' in avatar_data:
                self.avatar_name = avatar_data['name']
        
        if not self.avatar_name:
            print("Requesting current avatar info from VRChat...")
            self.vrchat_osc.request_avatar_info()
            # The response will come via the on_avatar_change callback
        else:
            print(f"Current avatar: {self.avatar_name}")
        return self.avatar_name
    
    def on_avatar_change(self, avatar_id):
        """Called when /avatar/change OSC event is received"""
        self.avatar_id = avatar_id
        print(f"Avatar changed to: {avatar_id}")
        
        # Update avatar name from the JSON file
        avatar_data = self.get_current_avatar()
        if avatar_data and 'name' in avatar_data:
            self.avatar_name = avatar_data['name']
            print(f"Avatar name: {self.avatar_name}")
    
    def on_avatar_parameter(self, *args):
        """Called when avatar parameter updates are received"""
        # This can help us detect when we're actively using an avatar
        if len(args) >= 1:
            value = args[0]
            # Parameter name would be in the OSC address path
            pass