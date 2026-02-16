from ultralytics import YOLO
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class BallTrackerNew:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_best_detection(self, frame):
        """ Получение лучших покадровых детекций мяча c использованием track """
        
        results = self.model.track(
            frame, 
            persist=True, 
            conf=0.6, 
            verbose=False, 
            tracker="trackers\\bytetrack_ball.yaml"
        )
        
        frame_detection = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                if len(boxes) > 0:
                    max_conf_idx = np.argmax(confidences)
                    best_box = boxes[max_conf_idx]
                    frame_detection = best_box.tolist()
        
        return frame_detection

    def get_positions_for_interpolation(self, detection_cache):
        """ Преобразование в формат для интерполяции """
        positions_for_interpolation = []
        
        for det in detection_cache:
            if len(det) == 4:  # Есть детекция [x1, y1, x2, y2]
                positions_for_interpolation.append(det)
            else:
                positions_for_interpolation.append([np.nan, np.nan, np.nan, np.nan])
        
        return positions_for_interpolation

    def interpolate_positions(self, positions, max_gap=10):
        """ Интерполяция пропущенных позиций мяча """
        if not positions:
            return positions
        
        df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        if df.isna().any().any():
            df_interpolated = df.interpolate(method='linear', limit=max_gap)
            df_filled = df_interpolated.ffill().bfill()
            positions = df_filled.to_numpy().tolist()
        
        return positions

    def smooth_trajectory(self, positions, window_size=3):
        """ Сглаживание траектории скользящим средним """
        if len(positions) < window_size:
            return positions
        
        positions_array = np.array(positions)
        smoothed = np.zeros_like(positions_array)
        
        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            
            smoothed[i] = np.mean(positions_array[start:end], axis=0)
        
        return smoothed.tolist()

    def get_all_extremums(self, ball_positions, debug=False):
        """Находит все экстремумы Y координаты"""
        y_values = []
        for ball_pos in ball_positions:
            if ball_pos and len(ball_pos) == 4:
                y1, y2 = ball_pos[1], ball_pos[3]
                center_y = (y1 + y2) / 2
                y_values.append(center_y)
            else:
                y_values.append(np.nan)
        
        y_array = np.array(y_values, dtype=np.float64)
        
        maxima_indices = argrelextrema(y_array, np.greater)[0]
        minima_indices = argrelextrema(y_array, np.less)[0]
        
        extremums = []
        
        for idx in maxima_indices:
            if 0 < idx < len(y_array):
                extremums.append({
                    'frame': int(idx),
                    'type': 'maximum',
                    'y_value': float(y_array[idx]),
                    'delta_y': None
                })
        
        for idx in minima_indices:
            if 0 < idx < len(y_array):
                extremums.append({
                    'frame': int(idx),
                    'type': 'minimum',
                    'y_value': float(y_array[idx]),
                    'delta_y': None
                })
        
        # Сортируем по кадрам
        extremums.sort(key=lambda x: x['frame'])
        
        if debug:
            print(f"Найдено {len(extremums)} экстремумов Y координаты: "
                f"{len([e for e in extremums if e['type']=='maximum'])} максимумов, "
                f"{len([e for e in extremums if e['type']=='minimum'])} минимумов")
        
        return extremums


