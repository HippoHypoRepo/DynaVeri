import cv2
import time
import argparse
from src.config_loader import load_config
from src.engine import PresenceEngine
from src.tracker import PresenceTracker
from src.visualizer import HUDVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Edge Presence Engine")
    parser.add_argument("--mode", type=str, choices=["text", "visual"])
    parser.add_argument("--items", type=str, help="Comma-separated target classes")
    parser.add_argument("--verify-mode", type=str, choices=["simultaneous", "latch"], help="Presence rule mode")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config("configs/workstation.yaml")
    
    if args.mode:
        config["engine"]["active_mode"] = args.mode
    if args.items and config["engine"]["active_mode"] == "text":
        config["engine"]["text_prompting"]["target_classes"] = [i.strip() for i in args.items.split(",")]
    if args.verify_mode:
        config["state_machine"]["verification_mode"] = args.verify_mode

    engine = PresenceEngine(config)
    tracker = PresenceTracker(config)
    visualizer = HUDVisualizer(config)
    
    cap = cv2.VideoCapture(config["system"]["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["system"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["system"]["frame_height"])

    print(f"[INFO] Engine initialized. Verification mode: {tracker.verification_mode.upper()}")
    print("       Press 't' to toggle rule mode (SIMULTANEOUS <-> LATCH)")
    print("       Press 'c' to clear/reset verification states")
    print("       Press 'v' to select ROI prompt (visual mode)")
    print("       Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        t_start = time.perf_counter()
        
        detections = engine.infer(frame)
        item_states, workspace_ready = tracker.update(detections)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        metrics = {"fps": 1000.0 / latency_ms if latency_ms > 0 else 0, "latency_ms": latency_ms}
        mode_info = {"verification_mode": tracker.verification_mode}
        
        out_frame = visualizer.render(frame, item_states, workspace_ready, metrics, mode_info)
        
        if config["engine"]["active_mode"] == "visual":
            cv2.putText(out_frame, "Press 'v' to select ROI prompt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        cv2.imshow("Edge Presence Engine", out_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            active_rule = tracker.toggle_mode()
            print(f"[INFO] Verification rule changed to: {active_rule.upper()}")
        elif key == ord('c'):
            tracker.reset()
        elif key == ord('r') and config["engine"]["active_mode"] == "text":
            new_config = load_config("configs/workstation.yaml")
            engine.reload_text_classes(new_config["engine"]["text_prompting"]["target_classes"])
            tracker.sync_classes(new_config["engine"]["text_prompting"]["target_classes"])
        elif key == ord('v') and config["engine"]["active_mode"] == "visual":
            roi = cv2.selectROI("Select Visual Prompt", frame, False, False)
            cv2.destroyWindow("Select Visual Prompt")
            if roi[2] > 0 and roi[3] > 0:
                x1, y1, w, h = roi
                engine.set_visual_prompt(frame.copy(), [x1, y1, x1 + w, y1 + h], "custom_object")
                tracker.sync_classes(["custom_object"])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()