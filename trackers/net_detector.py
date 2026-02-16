import numpy as np
from ultralytics import YOLO

class NetDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path)
    
    def detect_net_bbox(self, frame):
        if self.model is None:
            return None

        results = self.model(frame, verbose=False, classes=[2])
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                if confidences[best_idx] > 0.3:
                    box = boxes[best_idx].xyxy[0].cpu().numpy()
                    return box
        return None
    
    def get_net_line(self, frame, court_keypoints=None):
        net_bbox = self.detect_net_bbox(frame)
        
        if net_bbox is not None:
            x1, y1, x2, y2 = net_bbox
            net_y = y2
            net_line = [0, y2, frame.shape[1], y2]
            return net_line
        else:
            return None
