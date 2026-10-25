import cv2
import numpy as np

class HUDVisualizer:
    def __init__(self, config: dict):
        self.width = config["system"]["frame_width"]
        self.height = config["system"]["frame_height"]
        
        self.c_red = (40, 40, 255)
        self.c_yellow = (0, 215, 255)
        self.c_green = (50, 220, 50)
        self.c_bg = (20, 20, 20)
        self.c_text = (240, 240, 240)

    def render(self, frame: np.ndarray, item_states: dict, workspace_ready: bool, metrics: dict, mode_info: dict) -> np.ndarray:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        for item, data in item_states.items():
            if data["bbox"] is not None:
                x1, y1, x2, y2 = data["bbox"]
                color = self.c_green if data["state"] == "VERIFIED" else (self.c_yellow if data["state"] == "ACQUIRING" else self.c_red)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{item} {data['conf']:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        sidebar_w = 340
        sidebar_x = self.width - sidebar_w
        cv2.rectangle(frame, (sidebar_x, 0), (self.width, self.height), self.c_bg, -1)
        
        cv2.putText(frame, "WORKSPACE CHECKLIST", (sidebar_x + 15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, self.c_text, 1)
        
        # Display Active Rule Mode Badge
        rule_name = mode_info.get("verification_mode", "simultaneous").upper()
        rule_color = self.c_yellow if rule_name == "LATCH" else (200, 200, 200)
        cv2.putText(frame, f"RULE: {rule_name} (Toggle: 't')", (sidebar_x + 15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rule_color, 1)
        cv2.line(frame, (sidebar_x + 15, 68), (self.width - 15, 68), (100, 100, 100), 1)

        y_offset = 100
        for item, data in item_states.items():
            state = data["state"]
            
            if state == "VERIFIED":
                cv2.circle(frame, (sidebar_x + 30, y_offset - 5), 8, self.c_green, -1)
                status_text = "VERIFIED"
            elif state == "ACQUIRING":
                cv2.circle(frame, (sidebar_x + 30, y_offset - 5), 8, self.c_yellow, -1)
                status_text = "ACQUIRING"
            else:
                cv2.circle(frame, (sidebar_x + 30, y_offset - 5), 8, self.c_red, 2)
                status_text = "ABSENT"
                
            cv2.putText(frame, item.upper(), (sidebar_x + 55, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.c_text, 2)
            cv2.putText(frame, status_text, (sidebar_x + 55, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 55

        banner_y = self.height - 60
        if workspace_ready:
            cv2.rectangle(frame, (sidebar_x, banner_y), (self.width, self.height), self.c_green, -1)
            cv2.putText(frame, "WORKSPACE READY", (sidebar_x + 40, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(frame, (sidebar_x, banner_y), (self.width, self.height), self.c_red, -1)
            cv2.putText(frame, "INCOMPLETE", (sidebar_x + 85, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.c_text, 2)
            
        fps = metrics.get("fps", 0.0)
        latency = metrics.get("latency_ms", 0.0)
        cv2.rectangle(frame, (0, self.height - 40), (250, self.height), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency:.1f}ms", (10, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_yellow, 1)

        return frame