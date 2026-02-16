from ultralytics import YOLO
import pandas as pd
import numpy as np
from utils import measure_distance, get_center_of_bbox, get_foot_position
import cv2

class PlayerTrackerNew:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_player_detections(self, frame):
        """ Получение всех детекций игроков с track_id """
        
        results = self.model.track(
            frame, 
            persist=True, 
            conf=0.3,
            iou=0.5,
            verbose=False, 
            tracker="trackers\\bytetrack_player.yaml",
            classes=[2]
        )
        
        player_dict = {}
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else None
                confs = result.boxes.conf.cpu().numpy()
                
                if track_ids is not None:
                    for box, tid, conf in zip(boxes, track_ids, confs):
                        if conf > 0.3:
                            player_dict[int(tid)] = box.tolist()
        
        return player_dict
    
    def choose_players_for_frame(self, court_keypoints, player_dict, debug=False):
        """Выбор игроков для конкретного кадра"""
        UPPER_POINTS = [0, 4, 6, 1, 8, 12, 9] 
        LOWER_POINTS = [3, 5, 7, 2, 10, 13, 11]
        
        if not player_dict:
            return {}
        
        # Если нет валидных точек корта - используем первых двух игроков
        if court_keypoints is None or np.all(np.isnan(court_keypoints)):
            player_ids = list(player_dict.keys())
            result = {}
            for i, tid in enumerate(player_ids[:2]):
                result[tid] = 2 if i == 0 else 1  # Первый = верхний (2), второй = нижний (1)
            return result
        
        valid_points = []
        for i, kp in enumerate(court_keypoints):
            if i < len(court_keypoints) and not np.any(np.isnan(kp)):
                valid_points.append((i, kp))
        
        if len(valid_points) < 2:
            player_ids = list(player_dict.keys())
            result = {}
            for i, tid in enumerate(player_ids[:2]):
                result[tid] = 2 if i == 0 else 1
            return result
        
        upper_kps = [kp for i, kp in valid_points if i in UPPER_POINTS]
        lower_kps = [kp for i, kp in valid_points if i in LOWER_POINTS]
        
        # Если не хватает точек в группах, делим по медиане Y
        if len(upper_kps) == 0 or len(lower_kps) == 0:
            all_y = [kp[1] for _, kp in valid_points]
            median_y = np.median(all_y)
            upper_kps = [kp for _, kp in valid_points if kp[1] < median_y]
            lower_kps = [kp for _, kp in valid_points if kp[1] >= median_y]
        
        # Рассчитываем расстояния для каждого игрока
        player_distances = {}
        
        for track_id, bbox in player_dict.items():
            player_center = get_foot_position(bbox)
            
            upper_dist = min([measure_distance(player_center, kp) for kp in upper_kps]) if upper_kps else float('inf')
            
            lower_dist = min([measure_distance(player_center, kp) for kp in lower_kps]) if lower_kps else float('inf')
            
            player_distances[track_id] = {
                'upper_distance': upper_dist,
                'lower_distance': lower_dist,
                'is_closer_to_upper': upper_dist < lower_dist
            }
        
        chosen_players = {}
        
        num_players = len(player_dict)
        
        if num_players == 1:
            track_id = list(player_dict.keys())[0]
            distances = player_distances[track_id]
            
            if distances['is_closer_to_upper']:
                chosen_players[track_id] = 2
                if debug:
                    print(f"Кадр: 1 игрок, ближе к верхним точкам, назначаем ID 2")
            else:
                chosen_players[track_id] = 1
                if debug:
                    print(f"Кадр: 1 игрок, ближе к нижним точкам, назначаем ID 1")
                
        elif num_players >= 2:
            upper_candidates = [(tid, data['upper_distance']) for tid, data in player_distances.items()]
            upper_candidates.sort(key=lambda x: x[1])
            
            lower_candidates = [(tid, data['lower_distance']) for tid, data in player_distances.items()]
            lower_candidates.sort(key=lambda x: x[1])
            
            if upper_candidates:
                upper_player_id = upper_candidates[0][0]
                chosen_players[upper_player_id] = 2
            
            if lower_candidates:
                lower_player_id = lower_candidates[0][0]
                
                if lower_player_id in chosen_players:
                    for tid, _ in lower_candidates[1:]:
                        if tid not in chosen_players:
                            lower_player_id = tid
                            break
                
                chosen_players[lower_player_id] = 1
            
            # Если всё равно не нашли двух разных игроков
            if len(chosen_players) < 2:
                # Берем двух ближайших к корту игроков и назначаем ID на основе их позиции
                sorted_by_position = []
                for tid, data in player_distances.items():
                    position_score = data['upper_distance'] - data['lower_distance']
                    sorted_by_position.append((tid, position_score))
                
                sorted_by_position.sort(key=lambda x: x[1])  # Сортируем по position_score
                
                for i, (tid, score) in enumerate(sorted_by_position[:2]):
                    if tid not in chosen_players:
                        if score < 0:  # Ближе к верхним
                            chosen_players[tid] = 2
                        else:  # Ближе к нижним
                            chosen_players[tid] = 1
        
        return chosen_players
    
    def choose_and_filter_players_each_frame(self, court_keypoints, player_detections):
        """Новый метод: выбор игроков на каждом кадре отдельно"""
        if not isinstance(player_detections, list) or len(player_detections) == 0:
            return player_detections
        
        filtered_player_detections = []
        
        for frame_idx in range(len(player_detections)):
            current_player_dict = player_detections[frame_idx]
            
            current_court_kps = court_keypoints[frame_idx] if frame_idx < len(court_keypoints) else None
            
            chosen_players = self.choose_players_for_frame(current_court_kps, current_player_dict)
            
            filtered_player_dict = {}
            for track_id, assigned_id in chosen_players.items():
                if track_id in current_player_dict:
                    filtered_player_dict[assigned_id] = current_player_dict[track_id]
            
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections
        
    def draw_foot_positions(self, frame, foot_positions, player_colors=None):
        frame_out = frame.copy()
        
        if player_colors is None:
            player_colors = {
                1: (0, 255, 0),
                2: (255, 0, 0)
            }
        
        for player_id, foot_pos in foot_positions.items():
            if foot_pos is not None and len(foot_pos) >= 2:
                x, y = int(foot_pos[0]), int(foot_pos[1])
                color = player_colors.get(player_id, (0, 255, 255))
                
                cv2.circle(frame_out, (x, y), 12, color, -1)
                cv2.circle(frame_out, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(frame_out, f"Foot {player_id}", (x + 15, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_out, f"Foot {player_id}", (x + 15, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        return frame_out