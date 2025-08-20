import os
import json
from src.osc import VRChatOSC

class VRCAvatar:
    def __init__(self):
        self.vrchat_user_id = self._get_most_recent_user_id()
        
        self.vrchat_osc = VRChatOSC()
        self.avatar_id = "avtr_92982fd4-aedd-4adf-a5c6-e513b506dcd7"
        self.avatar_path = os.path.join(os.environ['USERPROFILE'], 'AppData', 'LocalLow', 'VRChat', 'VRChat', "OSC", self.vrchat_user_id, "Avatars")
    
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
        
        if not avatar_json_path:
            print("Error: No JSON files found in avatar directory")
            return None
        
        try:
            with open(avatar_json_path, 'r', encoding='utf-8-sig') as f:
                try:
                    d = json.load(f)
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

        # Set the avatar color using the VRChatOSC instance
        self.vrchat_osc.send_avatar_parameter(clothing_param_name, color)