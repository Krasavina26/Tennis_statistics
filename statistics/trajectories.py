import cv2
import numpy as np
import time
import math
import constants
from utils import convert_pixel_distance_to_meters

def create_fullscreen_mini_court_video(input_video_path, output_video_path, output_image_path,
                                     player_mini, ball_mini, mini_court_data, fps,
                                     player_avg_heights=None,
                                     trajectory_length=20, width=1280, height=720, debug=False):
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        if debug:
            print(f" Ошибка создания видеофайла: {output_video_path}")
        return
    
    player_colors = {
        1: (0, 255, 0),
        2: (255, 0, 0),
    }
    ball_color = (0, 255, 255)
    
    drawing_key_points = mini_court_data.drawing_key_points
    
    original_width = mini_court_data.court_drawing_width
    original_height = mini_court_data.court_drawing_height
    original_start_x = mini_court_data.court_start_x
    original_start_y = mini_court_data.court_start_y
    
    if player_avg_heights is None:
        player_avg_heights = {}
    
    scale_x = (width - 100) / original_width
    scale_y = (height - 100) / original_height
    scale = min(scale_x, scale_y) * 0.9
    
    # Размеры масштабированного корта (повернутого на 90°)
    new_width = int(original_height * scale)
    new_height = int(original_width * scale)
    offset_x = (width - new_width) // 2
    offset_y = (height - new_height) // 2
    
    player_trajectories = {1: [], 2: []}
    ball_trajectory = []
    all_player_trajectories = {1: [], 2: []}
    all_ball_trajectory = []
    player_distances_meters = {1: 0.0, 2: 0.0}
    ball_distance_meters = 0.0
    
    total_frames = min(len(player_mini), len(ball_mini))
    
    if debug:
        (f" Обработка {total_frames} кадров...")
    start_time = time.time()
    
    def calculate_player_distance_meters(point1, point2, player_id):
        """
        Рассчитывает расстояние в метрах для игрока с учетом его роста и средней высоты бокса
        """
        if point1 is None or point2 is None:
            return 0.0
        
        x1, y1 = point1
        x2, y2 = point2
        
        dx_pixels = x2 - x1
        dy_pixels = y2 - y1

        if player_id == 1:
            real_height_meters = constants.PLAYER_1_HEIGHT_METERS
        elif player_id == 2:
            real_height_meters = constants.PLAYER_2_HEIGHT_METERS
        else:
            real_height_meters = 1.75
        
        avg_bbox_height = player_avg_heights[player_id]
        
        dx_meters = convert_pixel_distance_to_meters(dx_pixels, real_height_meters, avg_bbox_height)
        dy_meters = convert_pixel_distance_to_meters(dy_pixels, real_height_meters, avg_bbox_height)
        
        distance = math.sqrt(dx_meters**2 + dy_meters**2)
        return distance
    
    def unscale_position(scaled_pos):
        """Преобразует масштабированные и повернутые координаты обратно в исходные координаты мини-корта"""
        if scaled_pos is None:
            return None
        
        try:
            x_scaled, y_scaled = scaled_pos
            
            x_rel = x_scaled - offset_x
            y_rel = y_scaled - offset_y
            
            orig_x = original_start_x + (new_height - y_rel) / scale
            orig_y = original_start_y + x_rel / scale
            
            return (orig_x, orig_y)
        except Exception as e:
            print(f"Ошибка в unscale_position: {e}")
            return None
    
    def rotate_and_scale_point(x, y):
        """
        Поворачивает точку на 90° по часовой стрелке и масштабирует
        """
        try:
            scaled_x = (x - original_start_x) * scale
            scaled_y = (y - original_start_y) * scale
            
            rotated_x = scaled_y
            rotated_y = original_width * scale - scaled_x
            
            final_x = int(rotated_x + offset_x)
            final_y = int(rotated_y + offset_y)
            
            return final_x, final_y
        except Exception as e:
            print(f"Ошибка в rotate_and_scale_point: {e}")
            return None, None
    
    def scale_position_with_rotation(pos):
        """Функция для масштабирования позиций с поворотом"""
        if pos is None:
            return None
        try:
            if isinstance(pos, (list, tuple, np.ndarray)):
                if len(pos) >= 2:
                    orig_x, orig_y = float(pos[0]), float(pos[1])
                else:
                    return None
            else:
                return None
            
            return rotate_and_scale_point(orig_x, orig_y)
        except Exception as e:
            print(f"Ошибка в scale_position_with_rotation: {e}")
            return None
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (255, 255, 255)
        
        # Рисуем фон
        bg_color = (229, 255, 204)
        cv2.rectangle(frame, 
                     (offset_x - 10, offset_y - 10),
                     (offset_x + new_width + 10, offset_y + new_height + 10),
                     bg_color, -1)
        
        # Рисуем линии корта с поворотом
        for line in mini_court_data.lines:
            start_idx, end_idx = line
            
            if start_idx * 2 + 1 < len(drawing_key_points) and end_idx * 2 + 1 < len(drawing_key_points):
                # Исходные координаты
                orig_x1 = drawing_key_points[start_idx * 2]
                orig_y1 = drawing_key_points[start_idx * 2 + 1]
                orig_x2 = drawing_key_points[end_idx * 2]
                orig_y2 = drawing_key_points[end_idx * 2 + 1]
                
                # Применяем поворот и масштабирование к обеим точкам
                x1, y1 = rotate_and_scale_point(orig_x1, orig_y1)
                x2, y2 = rotate_and_scale_point(orig_x2, orig_y2)
                
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Обновляем траектории
        current_player_pos = player_mini[frame_num] if frame_num < len(player_mini) else {}
        current_ball_pos = ball_mini[frame_num] if frame_num < len(ball_mini) else {}
        
        # Обработка игроков
        for player_id in [1, 2]:
            if player_id in current_player_pos:
                scaled_pos = scale_position_with_rotation(current_player_pos[player_id])
                if scaled_pos and scaled_pos[0] is not None and scaled_pos[1] is not None:
                    player_trajectories[player_id].append(scaled_pos)
                    all_player_trajectories[player_id].append(scaled_pos)
                    
                    if len(player_trajectories[player_id]) >= 2:
                        prev_scaled = player_trajectories[player_id][-2]
                        curr_scaled = player_trajectories[player_id][-1]
                        
                        prev_original = unscale_position(prev_scaled)
                        curr_original = unscale_position(curr_scaled)
                        
                        if prev_original and curr_original:
                            distance_segment = calculate_player_distance_meters(prev_original, curr_original, player_id)
                            player_distances_meters[player_id] += distance_segment
                    
                    if len(player_trajectories[player_id]) > trajectory_length:
                        player_trajectories[player_id].pop(0)
        
        # Обработка мяча
        if 1 in current_ball_pos:
            scaled_ball_pos = scale_position_with_rotation(current_ball_pos[1])
            if scaled_ball_pos and scaled_ball_pos[0] is not None and scaled_ball_pos[1] is not None:
                ball_trajectory.append(scaled_ball_pos)
                all_ball_trajectory.append(scaled_ball_pos)
                
                if len(ball_trajectory) > trajectory_length * 2:
                    ball_trajectory.pop(0)
        
        # Рисуем траектории игроков
        for player_id in [1, 2]:
            trajectory = player_trajectories[player_id]
            if len(trajectory) > 1:
                color = player_colors[player_id]
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i-1]
                    pt2 = trajectory[i]
                    if pt1 and pt2 and pt1[0] is not None and pt1[1] is not None and pt2[0] is not None and pt2[1] is not None:
                        alpha = 0.3 + 0.7 * (i / len(trajectory))
                        color_with_alpha = tuple(int(c * alpha) for c in color)
                        cv2.line(frame, pt1, pt2, color_with_alpha, 3)
        
        # Рисуем траекторию мяча
        if len(ball_trajectory) > 1:
            for i in range(1, len(ball_trajectory)):
                pt1 = ball_trajectory[i-1]
                pt2 = ball_trajectory[i]
                if pt1 and pt2 and pt1[0] is not None and pt1[1] is not None and pt2[0] is not None and pt2[1] is not None:
                    alpha = 0.3 + 0.7 * (i / len(ball_trajectory))
                    color_with_alpha = tuple(int(c * alpha) for c in ball_color)
                    cv2.line(frame, pt1, pt2, color_with_alpha, 4)
        
        # Рисуем текущие позиции игроков
        for player_id, pos in current_player_pos.items():
            scaled_pos = scale_position_with_rotation(pos)
            if scaled_pos and scaled_pos[0] is not None and scaled_pos[1] is not None:
                color = player_colors.get(player_id, (255, 255, 255))
                x, y = scaled_pos
                cv2.circle(frame, (x, y), 10, color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(frame, f"P{player_id}", (x + 12, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
                cv2.putText(frame, f"P{player_id}", (x + 12, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Рисуем текущую позицию мяча
        if 1 in current_ball_pos:
            scaled_ball_pos = scale_position_with_rotation(current_ball_pos[1])
            if scaled_ball_pos and scaled_ball_pos[0] is not None and scaled_ball_pos[1] is not None:
                x, y = scaled_ball_pos
                cv2.circle(frame, (x, y), 8, ball_color, -1)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(frame, "BALL", (x + 12, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(frame, "BALL", (x + 12, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
        
        out.write(frame)
        
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            fps_curr = (frame_num + 1) / elapsed if elapsed > 0 else 0
            if debug:
                print(f"  Обработано: {frame_num}/{total_frames} кадров ({fps_curr:.1f} FPS)")
    
    out.release()
    
    save_fullscreen_summary_image(output_image_path, width, height,
                                 all_player_trajectories,
                                 player_colors,
                                 drawing_key_points, original_start_x, original_start_y,
                                 scale, offset_x, offset_y, mini_court_data,
                                 player_distances_meters)
    
    return {
        'player_distances': player_distances_meters,
        'total_frames': total_frames,
        'total_time_seconds': total_frames / fps if fps > 0 else 0
    }

def save_fullscreen_summary_image(output_path, width, height,
                                 player_trajectories,
                                 player_colors,
                                 drawing_key_points, orig_start_x, orig_start_y,
                                 scale, offset_x, offset_y, mini_court,
                                 player_distances=None):
    """
    Сохраняет итоговое изображение с траекториями (упрощенная версия)
    """
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (255, 255, 255)
    
    bg_color = (204, 255, 204)
    
    original_width = mini_court.court_drawing_width
    original_height = mini_court.court_drawing_height
    
    new_width = int(original_height * scale)
    new_height = int(original_width * scale)
    
    cv2.rectangle(frame, 
                 (offset_x - 10, offset_y - 10),
                 (offset_x + new_width + 10, offset_y + new_height + 10),
                 bg_color, -1)
    
    def rotate_and_scale_point(x, y):
        scaled_x = (x - orig_start_x) * scale
        scaled_y = (y - orig_start_y) * scale
        
        rotated_x = scaled_y
        rotated_y = original_width * scale - scaled_x
        
        final_x = int(rotated_x + offset_x)
        final_y = int(rotated_y + offset_y)
        
        return final_x, final_y
    
    # Рисуем линии корта
    for line in mini_court.lines:
        start_idx, end_idx = line
        
        if start_idx * 2 + 1 < len(drawing_key_points) and end_idx * 2 + 1 < len(drawing_key_points):
            orig_x1 = drawing_key_points[start_idx * 2]
            orig_y1 = drawing_key_points[start_idx * 2 + 1]
            orig_x2 = drawing_key_points[end_idx * 2]
            orig_y2 = drawing_key_points[end_idx * 2 + 1]
            
            x1, y1 = rotate_and_scale_point(orig_x1, orig_y1)
            x2, y2 = rotate_and_scale_point(orig_x2, orig_y2)
            
            cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), 3)
    
    # Рисуем траектории игроков
    for player_id in [1, 2]:
        trajectory = player_trajectories.get(player_id, [])
        if len(trajectory) > 1:
            color = player_colors[player_id]
            
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i-1]
                pt2 = trajectory[i]
                
                if pt1 and pt2:
                    alpha = 0.3 + 0.7 * (i / len(trajectory))
                    color_with_alpha = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pt1, pt2, color_with_alpha, 2)
    
    cv2.imwrite(output_path, frame)