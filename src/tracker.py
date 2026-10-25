class PresenceState:
    ABSENT = "ABSENT"
    ACQUIRING = "ACQUIRING"
    VERIFIED = "VERIFIED"
    DECAYING = "DECAYING"

class PresenceTracker:
    def __init__(self, config: dict):
        self.config = config
        self.mode = self.config["engine"]["active_mode"]
        
        if self.mode == "text":
            self.target_classes = self.config["engine"]["text_prompting"]["target_classes"]
        else:
            exemplars = self.config["engine"]["visual_prompting"].get("exemplars", [])
            self.target_classes = [exemplars[0]["name"]] if exemplars else ["custom_object"]
            
        self.verification_mode = self.config["state_machine"]["verification_mode"]
        self.debounce_limit = self.config["state_machine"]["debounce_frames"]
        self.decay_limit = self.config["state_machine"]["decay_frames"]
        
        self.item_states = {}
        self.reset()
        
    def reset(self):
        for cls_name in self.target_classes:
            self.item_states[cls_name] = {
                "state": PresenceState.ABSENT,
                "hit_count": 0,
                "miss_count": 0,
                "bbox": None,
                "conf": 0.0
            }

    def toggle_mode(self) -> str:
        if self.verification_mode == "simultaneous":
            self.verification_mode = "latch"
        else:
            self.verification_mode = "simultaneous"
            # Invalidate missing items when switching back to simultaneous
            for data in self.item_states.values():
                if data["bbox"] is None:
                    data["state"] = PresenceState.ABSENT
        return self.verification_mode
            
    def sync_classes(self, new_classes: list):
        self.target_classes = new_classes
        current_keys = list(self.item_states.keys())
        
        for key in current_keys:
            if key not in new_classes:
                del self.item_states[key]
                
        for cls_name in new_classes:
            if cls_name not in self.item_states:
                self.item_states[cls_name] = {
                    "state": PresenceState.ABSENT,
                    "hit_count": 0,
                    "miss_count": 0,
                    "bbox": None,
                    "conf": 0.0
                }

    def update(self, detections: list) -> tuple[dict, bool]:
        detected_this_frame = {}
        for det in detections:
            cls_name = det["class_name"]
            if cls_name in self.target_classes:
                if cls_name not in detected_this_frame or det["conf"] > detected_this_frame[cls_name]["conf"]:
                    detected_this_frame[cls_name] = det

        for cls_name in self.target_classes:
            state_info = self.item_states[cls_name]
            
            if cls_name in detected_this_frame:
                det = detected_this_frame[cls_name]
                state_info["bbox"] = det["bbox"]
                state_info["conf"] = det["conf"]
                state_info["miss_count"] = 0
                
                if state_info["state"] in (PresenceState.ABSENT, PresenceState.DECAYING):
                    state_info["state"] = PresenceState.ACQUIRING
                    
                if state_info["state"] == PresenceState.ACQUIRING:
                    state_info["hit_count"] += 1
                    if state_info["hit_count"] >= self.debounce_limit:
                        state_info["state"] = PresenceState.VERIFIED
                        
            else:
                if self.verification_mode == "latch" and state_info["state"] == PresenceState.VERIFIED:
                    state_info["bbox"] = None
                    state_info["conf"] = 0.0
                    continue

                state_info["hit_count"] = 0
                state_info["miss_count"] += 1
                
                if state_info["state"] == PresenceState.VERIFIED:
                    state_info["state"] = PresenceState.DECAYING
                    
                if state_info["state"] == PresenceState.DECAYING and state_info["miss_count"] >= self.decay_limit:
                    state_info["state"] = PresenceState.ABSENT
                    state_info["bbox"] = None
                    state_info["conf"] = 0.0
                    
                if state_info["state"] == PresenceState.ACQUIRING:
                    state_info["state"] = PresenceState.ABSENT
                    state_info["bbox"] = None
                    state_info["conf"] = 0.0

        workspace_ready = all(
            info["state"] == PresenceState.VERIFIED 
            for info in self.item_states.values()
        )
        
        return self.item_states, workspace_ready