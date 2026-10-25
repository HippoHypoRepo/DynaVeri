import cv2
import numpy as np
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

class PresenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.mode = self.config["engine"]["active_mode"]
        self.device = self.config["system"]["device"]
        
        if self.mode == "text":
            self._init_text_model()
        elif self.mode == "visual":
            self._init_visual_model()
        else:
            raise ValueError(f"Unknown active_mode: {self.mode}")

    def _pre_letterbox(self, image: np.ndarray, bbox: list, target_size: int = 640) -> tuple:
        """
        Optimizes the reference image ONCE.
        Resizes and pads the image to exactly 640x640 and scales the bounding box mathematically.
        This completely bypasses the Ultralytics CPU pre-processing bottleneck inside the loop.
        """
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Calculate padding to center the image
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        
        # Create a 640x640 gray canvas (Ultralytics default pad color is 114)
        padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Scale and shift the bounding box to match the new padded canvas
        x1, y1, x2, y2 = bbox
        new_bbox = [
            x1 * scale + pad_w,
            y1 * scale + pad_h,
            x2 * scale + pad_w,
            y2 * scale + pad_h
        ]
        
        return padded_img, new_bbox

    def _init_text_model(self):
        text_cfg = self.config["engine"]["text_prompting"]
        self.model = YOLO(text_cfg["model_checkpoint"]).to(self.device)
        self.target_classes = text_cfg["target_classes"]
        self.conf = text_cfg["confidence_threshold"]
        self.iou = text_cfg["iou_threshold"]
        self.model.set_classes(self.target_classes)

    def _init_visual_model(self):
        vis_cfg = self.config["engine"]["visual_prompting"]
        self.model = YOLOE(vis_cfg["model_checkpoint"]).to(self.device)
        self.conf = vis_cfg["confidence_threshold"]
        self.iou = vis_cfg["iou_threshold"]
        
        self.target_classes = []
        self.refer_image = None
        self.visual_prompts = {
            "bboxes": np.empty((0, 4), dtype=np.float32), 
            "cls": np.empty((0,), dtype=np.int32)
        }
        
        if vis_cfg.get("exemplars"):
            exemplar = vis_cfg["exemplars"][0]
            self.target_classes.append(exemplar["name"])
            img = cv2.imread(exemplar["image_path"])
            if img is not None:
                # Statically downscale config exemplar to 320x320
                orig_h, orig_w = img.shape[:2]
                target_size = 320
                scale_x, scale_y = target_size / orig_w, target_size / orig_h
                
                self.refer_image = cv2.resize(img, (target_size, target_size))
                x1, y1, x2, y2 = exemplar["bbox"]
                scaled_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                
                self.visual_prompts = {
                    "bboxes": np.array([scaled_bbox], dtype=np.float32),
                    "cls": np.array([0], dtype=np.int32)
                }

    def set_visual_prompt(self, image: np.ndarray, bbox: list, class_name: str):
        self.target_classes = [class_name]
        
        # 1. Capture original dimensions of the live frame
        orig_h, orig_w = image.shape[:2]
        
        # 2. Define a lightweight target resolution for the reference encoder
        target_size = 320 
        
        # 3. Calculate exact scaling factors
        scale_x = target_size / orig_w
        scale_y = target_size / orig_h
        
        # 4. Resize the reference image permanently in memory (Massive CPU/PCIe save)
        self.refer_image = cv2.resize(image, (target_size, target_size))
        
        # 5. Mathematically scale the user's ROI to match the new 320x320 reference
        x1, y1, x2, y2 = bbox
        scaled_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        
        self.visual_prompts = {
            "bboxes": np.array([scaled_bbox], dtype=np.float32),
            "cls": np.array([0], dtype=np.int32)
        }

    def infer(self, frame: np.ndarray) -> list:
        if self.mode == "text":
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )
        elif self.mode == "visual":
            if self.refer_image is None or self.refer_image.size == 0:
                return []
                
            results = self.model.predict(
                source=frame,
                device=self.device,
                predictor=YOLOEVPSegPredictor,
                refer_image=self.refer_image,
                visual_prompts=self.visual_prompts,
                conf=self.conf,
                iou=self.iou,
                imgsz=320,
                max_det=5,
                verbose=False
            )
            
        return self._parse_results(results)

    def reload_text_classes(self, new_classes: list):
        if self.mode == "text" and new_classes != self.target_classes:
            self.target_classes = new_classes
            self.model.set_classes(self.target_classes)

    def _parse_results(self, results) -> list:
        detections = []
        if not results or len(results) == 0:
            return detections
            
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                class_name = self.target_classes[cls_id] if cls_id < len(self.target_classes) else f"unknown_{cls_id}"
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "class_name": class_name
                })
                
        return detections
