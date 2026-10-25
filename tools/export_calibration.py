import cv2
import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import load_config

def main():
    config = load_config("configs/workstation.yaml")
    out_dir = "assets/calibration_1024"
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(config["system"]["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["system"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["system"]["frame_height"])
    
    count = 0
    print(f"[INFO] Collecting 1,024 calibration frames to {out_dir}...")
    print("       Press 's' to manually save a frame.")
    print("       Press 'a' to toggle auto-capture (1 frame per 10 ticks).")
    print("       Press 'q' to quit.")
    
    auto_mode = False
    tick = 0
    
    while count < 1024:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        mode_text = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(display, f"Captured: {count}/1024 | Mode: {mode_text}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
        
        cv2.imshow("Hailo INT8 Calibration Exporter", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
            
        tick += 1
        if key == ord('s') or (auto_mode and tick % 10 == 0):
            cv2.imwrite(os.path.join(out_dir, f"frame_{count:04d}.jpg"), frame)
            count += 1
            
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Calibration dataset collection complete.")

if __name__ == "__main__":
    main()