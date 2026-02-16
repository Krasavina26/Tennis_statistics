import constants
import cv2
from collections import defaultdict
import math
from utils import convert_pixel_distance_to_meters, get_height_of_bbox
import constants

class SpeedCalculator:
    def __init__(self, fps, court_width_meters=constants.DOUBLE_LINE_WIDTH, court_length_meters=constants.COURT_HEIGHT):
        self.fps = fps
        self.court_width_meters = court_width_meters
        self.court_length_meters = court_length_meters
        self.player_positions_history = defaultdict(list)
        
        self.player_avg_heights = {}
        self.player_heights_samples = defaultdict(list)
        
        self.pixels_to_meters_x = None
        self.pixels_to_meters_y = None
        
        self.PANEL_WIDTH_RATIO = 0.15
        self.PANEL_HEIGHT_RATIO = 0.1
        self.MIN_PANEL_WIDTH = 150
        self.MIN_PANEL_HEIGHT = 100
        self.MAX_PANEL_WIDTH = 400
        self.MAX_PANEL_HEIGHT = 200
        
        self.MARGIN_RIGHT = 20
        self.MARGIN_BOTTOM = 20
        self.PADDING = 15
        self.TEXT_LINE_HEIGHT = 30
    
    def update_player_avg_height(self, player_id, bbox):
        """
        Обновляет среднюю высоту бокса для игрока
        """
        if bbox is None:
            return
        
        height = get_height_of_bbox(bbox)
        self.player_heights_samples[player_id].append(height)
        
        max_samples = 30
        if len(self.player_heights_samples[player_id]) > max_samples:
            self.player_heights_samples[player_id].pop(0)
        
        if self.player_heights_samples[player_id]:
            self.player_avg_heights[player_id] = sum(self.player_heights_samples[player_id]) / len(self.player_heights_samples[player_id])
    
    def calculate_player_speed_from_mini_coords(self, current_pos, previous_pos, player_id, time_interval_frames=5):
        """
        Рассчитывает скорость игрока на основе координат мини-корта
        Использует среднюю высоту бокса игрока и его реальный рост
        """
        if current_pos is None or previous_pos is None:
            return 0, 0
        
        dx_pixels = current_pos[0] - previous_pos[0]
        dy_pixels = current_pos[1] - previous_pos[1]
        
        avg_bbox_height = self.player_avg_heights[player_id]
        
        if player_id == 1:
            real_height_meters = constants.PLAYER_1_HEIGHT_METERS
        elif player_id == 2:
            real_height_meters = constants.PLAYER_2_HEIGHT_METERS
        else:
            real_height_meters = 1.75
        
        dx_meters = convert_pixel_distance_to_meters(dx_pixels, real_height_meters, avg_bbox_height)
        dy_meters = convert_pixel_distance_to_meters(dy_pixels, real_height_meters, avg_bbox_height)
        
        distance_meters = math.sqrt(dx_meters**2 + dy_meters**2)
        time_seconds = time_interval_frames / self.fps
        
        if time_seconds > 0:
            speed_m_s = distance_meters / time_seconds
        else:
            speed_m_s = 0
        
        speed_km_h = speed_m_s * 3.6
        
        return speed_m_s, speed_km_h
    
    def update_player_speed(self, player_id, player_position, player_bbox=None):
        """
        Обновляет и рассчитывает скорость игрока
        """
        if player_position is None:
            return 0, 0
        
        if player_bbox is not None:
            self.update_player_avg_height(player_id, player_bbox)

        self.player_positions_history[player_id].append(player_position)
        
        max_history = 15
        if len(self.player_positions_history[player_id]) > max_history:
            self.player_positions_history[player_id].pop(0)
        
        if len(self.player_positions_history[player_id]) < 5:
            return 0, 0
        
        interval = min(5, len(self.player_positions_history[player_id]) - 1)
        current_pos = self.player_positions_history[player_id][-1]
        previous_pos = self.player_positions_history[player_id][-interval]
        
        speed_m_s, speed_km_h = self.calculate_player_speed_from_mini_coords(
            current_pos, previous_pos, player_id, interval
        )
        
        return speed_m_s, speed_km_h

    def draw_speed_info(self, frame, player_speeds, position=None):
        """
        Рисует информацию о скорости в нижнем правом углу кадра
        """
        if frame is None:
            return frame
        
        frame_out = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        panel_width, panel_height = self._calculate_panel_size(frame_width, frame_height)
        
        panel_x = frame_width - panel_width - self.MARGIN_RIGHT
        panel_y = frame_height - panel_height - self.MARGIN_BOTTOM
        
        panel_x = max(self.PADDING, panel_x)
        panel_y = max(self.PADDING, panel_y)
        
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame_out, 1 - alpha, 0, frame_out)
        
        cv2.rectangle(frame_out, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (200, 200, 200), 2)
        
        text_x = panel_x + self.PADDING
        text_y = panel_y + self.PADDING + 25
        
        title_font_scale = min(0.8, panel_width / 400)
        text_font_scale = min(0.6, panel_width / 450)
        small_font_scale = min(0.5, panel_width / 500)
        title_thickness = 2
        text_thickness = max(1, int(panel_width / 250))
        
        title = "SPEED METER"
        cv2.putText(frame_out, title, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, 
                   (255, 255, 255), title_thickness)
        
        text_y += int(self.TEXT_LINE_HEIGHT * 0.9)
        
        players_title = "PLAYERS:"
        cv2.putText(frame_out, players_title, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, 
                   (220, 220, 220), text_thickness)
        
        text_y += int(self.TEXT_LINE_HEIGHT * 0.8)
        
        # Отображаем скорости игроков
        for player_id, speed in sorted(player_speeds.items()):
            if player_id == 1:
                player_color = (131, 248, 255)
            elif player_id == 2:
                player_color = (255, 131, 135)
            else:
                player_color = (255, 0, 0)
            
            player_text = f"P{player_id}: {speed:5.1f} km/h"
            cv2.putText(frame_out, player_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, 
                       player_color, text_thickness)
            
            bar_width_small = int(panel_width * 0.4)
            bar_height_small = 6
            bar_x_small = text_x + int(panel_width * 0.55)
            
            cv2.rectangle(frame_out, (bar_x_small, text_y - 8),
                         (bar_x_small + bar_width_small, text_y - 8 + bar_height_small),
                         (70, 70, 70), -1)
            
            max_player_speed = 25
            fill_width_small = min(int((speed / max_player_speed) * bar_width_small), 
                                  bar_width_small)
            cv2.rectangle(frame_out, (bar_x_small, text_y - 8),
                         (bar_x_small + fill_width_small, text_y - 8 + bar_height_small),
                         player_color, -1)
            
            cv2.rectangle(frame_out, (bar_x_small, text_y - 8),
                         (bar_x_small + bar_width_small, text_y - 8 + bar_height_small),
                         (180, 180, 180), 1)
            
            text_y += int(self.TEXT_LINE_HEIGHT * 0.7)
        
        return frame_out
    
    def _calculate_panel_size(self, frame_width, frame_height):
        """Вычисляет размер панели в зависимости от размера кадра"""
        panel_width = int(frame_width * self.PANEL_WIDTH_RATIO)
        panel_height = int(frame_height * self.PANEL_HEIGHT_RATIO)
        
        panel_width = max(self.MIN_PANEL_WIDTH, min(panel_width, self.MAX_PANEL_WIDTH))
        panel_height = max(self.MIN_PANEL_HEIGHT, min(panel_height, self.MAX_PANEL_HEIGHT))
        
        lines_needed = 5
        panel_height = max(panel_height, lines_needed * 25 + 40)
        
        return panel_width, panel_height