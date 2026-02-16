import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    get_foot_position,
    get_center_of_bbox
)
from trackers import NetDetector
from court_line_detector import CourtAreaAnalyzerWithNet

class MiniCourt():
    def __init__(self, frame):
        self.video_height, self.video_width = frame.shape[:2]
        
        base_width = min(200, self.video_width // 8)
        base_width = max(base_width, 200)
        
        self.drawing_rectangle_width = base_width
        self.drawing_rectangle_height = int(base_width * 2)
        
        if self.drawing_rectangle_height > self.video_height * 0.7:
            self.drawing_rectangle_height = int(self.video_height * 0.7)
            self.drawing_rectangle_width = int(self.drawing_rectangle_height / 2)
        
        self.buffer = 10
        
        self.padding_court = int(min(self.drawing_rectangle_width, self.drawing_rectangle_height) * 0.08)  # 8% от размера

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        
        self.homography_matrix = None
        self.source_points = None  # Точки на основном видео
        self.destination_points = None  # Точки на мини-корте

    def set_homography(self, source_points, destination_points, debug=False):
        """Устанавливает матрицу гомографии на основе соответствий точек"""
        if len(source_points) < 4 or len(destination_points) < 4:
            if debug:
                print(f"Предупреждение: нужно минимум 4 точки для гомографии. Получено: {len(source_points)}")
            return
        
        src_pts = np.array(source_points, dtype=np.float32)
        dst_pts = np.array(destination_points, dtype=np.float32)
        
        self.homography_matrix, mask = cv2.findHomography(src_pts, dst_pts)
        
        if self.homography_matrix is None:
            if debug:
                print("Ошибка: не удалось вычислить матрицу гомографии")
        else:
            self.source_points = src_pts
            self.destination_points = dst_pts
            if debug:
                print(f"Гомография установлена успешно. Матрица shape: {self.homography_matrix.shape}")
    
    def convert_to_mini_court_with_homography(self, point, debug=False):
        """Преобразует точку с основного видео на мини-корт с помощью гомографии"""
        if self.homography_matrix is None:
            if debug:
                print("Предупреждение: гомография не установлена")
            return None
        
        if point is None or np.any(np.isnan(point)):
            return None
        
        try:
            point_array = np.array([[point[0], point[1]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), self.homography_matrix)
            
            if transformed is not None and len(transformed) > 0:
                x_mini = float(transformed[0][0][0])
                y_mini = float(transformed[0][0][1])
                return (x_mini, y_mini)
        except Exception as e:
            print(f"Ошибка при преобразовании точки {point}: {e}")
        
        return None
    
    def initialize_homography_from_court_keypoints(self, court_keypoints_video, debug=False):
        """Автоматическая инициализация гомографии на основе ключевых точек корта"""
        if court_keypoints_video is None:
            if debug:
                print("Ключевые точки корта не предоставлены")
            return
        
        video_points = []
        mini_points = []
        
        corner_indices = [0, 1, 2, 3]
        
        for idx in corner_indices:
            if isinstance(court_keypoints_video, list):
                if idx * 2 + 1 < len(court_keypoints_video):
                    x_video = court_keypoints_video[idx * 2]
                    y_video = court_keypoints_video[idx * 2 + 1]
            elif isinstance(court_keypoints_video, np.ndarray):
                if court_keypoints_video.shape[0] > idx:
                    x_video = court_keypoints_video[idx, 0]
                    y_video = court_keypoints_video[idx, 1]
            else:
                if debug:
                    print(f"Неизвестный формат ключевых точек: {type(court_keypoints_video)}")
                return
            
            if idx * 2 + 1 < len(self.drawing_key_points):
                x_mini = self.drawing_key_points[idx * 2]
                y_mini = self.drawing_key_points[idx * 2 + 1]
                
                video_points.append([float(x_video), float(y_video)])
                mini_points.append([float(x_mini), float(y_mini)])
        
        if len(video_points) >= 4 and len(mini_points) >= 4:
            self.set_homography(video_points, mini_points)
            if debug:
                print(f"Гомография инициализирована с {len(video_points)} точками")
        else:
            if debug:
                print(f"Не удалось собрать достаточное количество точек: video={len(video_points)}, mini={len(mini_points)}")
    
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*38
        # point 0
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 3
        drawing_key_points[6] = int(self.court_start_x)
        drawing_key_points[7] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 2
        drawing_key_points[4] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[5] = drawing_key_points[7] 
        # point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # point 5
        drawing_key_points[10] = drawing_key_points[6] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[7] 
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # point 7
        drawing_key_points[14] = drawing_key_points[4] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[5] 
        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21]

        # point 14
        drawing_key_points[28] = drawing_key_points[0]
        drawing_key_points[29] = int(drawing_key_points[1] + drawing_key_points[5])/2

        # point 15
        drawing_key_points[30] = drawing_key_points[8]
        drawing_key_points[31] = drawing_key_points[29]

        # point 16
        drawing_key_points[32] = drawing_key_points[24]
        drawing_key_points[33] = drawing_key_points[29]

        # point 17
        drawing_key_points[34] = drawing_key_points[14]
        drawing_key_points[35] = drawing_key_points[29]

        # point 18
        drawing_key_points[36] = drawing_key_points[2]
        drawing_key_points[37] = drawing_key_points[29]

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 3),
            (4, 5),
            (6,7),
            (1,2),
            
            (0,1),
            (8,9),
            (10,11),
            (2,3),
            (12, 13),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def set_canvas_background_box_position(self,frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        # for i in range(0, len(self.drawing_key_points),2):
        #     x = int(self.drawing_key_points[i])
        #     y = int(self.drawing_key_points[i+1])
        #     cv2.circle(frame, (x,y), 5, (0,0,255),-1)

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (0, 0, 255), 3)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.6
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def draw_points_on_mini_court(self, frame, positions, color=(0,255,0), radius=5):
        if positions is None or not positions:
            return frame

        frame_out = frame.copy()
        for _, position in positions.items():
            try:
                pos_array = np.array(position)
                
                # Если это 2D массив [[x, y]], берем первый элемент
                if len(pos_array.shape) > 1:
                    x = float(pos_array[0, 0])
                    y = float(pos_array[0, 1])
                else:
                    # Если это одномерный массив [x, y]
                    if len(pos_array) >= 2:
                        x = float(pos_array[0])
                        y = float(pos_array[1])
                    else:
                        continue
                        
                if np.isnan(x) or np.isnan(y):
                    continue
                    
                x = int(x)
                y = int(y)

                x, y = self._clamp_to_mini_court((x, y))
                
                cv2.circle(frame_out, (x, y), radius, color, -1)
                    
            except (TypeError, ValueError, IndexError) as e:
                continue
        
        return frame_out

    def _clamp_to_mini_court(self, point, margin=20):
        """
        Ограничивает точку в пределах мини-корта с небольшим отступом
        """
        x, y = point
        
        min_x = self.court_start_x - margin
        max_x = self.court_end_x + margin
        min_y = self.court_start_y - margin
        max_y = self.court_end_y + margin
        
        clamped_x = max(min_x, min(x, max_x))
        clamped_y = max(min_y, min(y, max_y))
        
        return (clamped_x, clamped_y)

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, court_keypoints_video, debug=False):
        """
        Основной метод конвертации через гомографию
        """
        if self.homography_matrix is None and court_keypoints_video is not None:
            self.initialize_homography_from_court_keypoints(court_keypoints_video)
        
        output_player_boxes = []
        output_ball_boxes = []
        
        total_frames = min(len(player_boxes), len(ball_boxes))
        
        for frame_num in range(total_frames):
            player_mini_positions = {}
            if frame_num < len(player_boxes):
                for player_id, bbox in player_boxes[frame_num].items():
                    if bbox is None:
                        continue
                        
                    foot_position = get_foot_position(bbox)
                    if foot_position is None or np.any(np.isnan(foot_position)):
                        continue
                    
                    mini_coords = self.convert_to_mini_court_with_homography(foot_position)
                    
                    if mini_coords is not None:
                        mini_coords = self._clamp_to_mini_court(mini_coords)
                        player_mini_positions[player_id] = mini_coords
            
            output_player_boxes.append(player_mini_positions)
            
            # Конвертируем позицию мяча
            ball_mini_position = {}
            if frame_num < len(ball_boxes):
                ball_box = ball_boxes[frame_num]
                if ball_box is not None and not np.any(np.isnan(ball_box)):
                    ball_center = get_center_of_bbox(ball_box)
                    if ball_center is not None:
                        mini_coords = self.convert_to_mini_court_with_homography(ball_center)
                        if mini_coords is not None:
                            mini_coords = self._clamp_to_mini_court(mini_coords)
                            ball_mini_position[1] = mini_coords
            
            output_ball_boxes.append(ball_mini_position)
            
            if frame_num % 100 == 0:
                if debug:
                    print(f"   Конвертация через гомографию: {frame_num}/{total_frames} кадров")
        
        return output_player_boxes, output_ball_boxes
    
class MiniCourtWithNetDivision(MiniCourt):
    """MiniCourt с разделением зон сеткой и соответствием зон основному корту"""
    
    def __init__(self, frame, net_detector_path=None, fullscreen_mode=False):
        self.fullscreen_mode = fullscreen_mode
        
        if fullscreen_mode:
            self.drawing_rectangle_width = frame.shape[1] - 100
            self.drawing_rectangle_height = frame.shape[0] - 100
            self.buffer = 50  # Минимальный отступ
        else:
            self.video_height, self.video_width = frame.shape[:2]
        
            base_width = min(300, self.video_width // 5)
            base_width = max(base_width, 150)
            
            self.drawing_rectangle_width = base_width
            self.drawing_rectangle_height = int(base_width * 2)
            
            if self.drawing_rectangle_height > self.video_height * 0.7:
                self.drawing_rectangle_height = int(self.video_height * 0.7)
                self.drawing_rectangle_width = int(self.drawing_rectangle_height / 2)
            
            self.buffer = 50+80
            
        self.padding_court = int(min(self.drawing_rectangle_width, self.drawing_rectangle_height) * 0.08)
        
        super().__init__(frame)
        
        self.real_court_width = constants.DOUBLE_LINE_WIDTH
        self.net_detector = NetDetector(net_detector_path)
        self.area_analyzer = CourtAreaAnalyzerWithNet()
        self.net_line = None

        self.mini_areas_with_zones = {}
        
        self.zone_colors = {
            'left_doubles_area_upper': (200, 255, 200),
            'left_doubles_area_lower': (255, 150, 150),
            'right_doubles_area_upper': (255, 200, 200),
            'right_doubles_area_lower': (150, 255, 150),
            'upper_baseline_area': (100, 100, 255),
            'lower_baseline_area': (100, 100, 255),
            'right_center_area_upper': (255, 255, 200),
            'right_center_area_lower': (150, 255, 255),
            'left_center_area_upper': (200, 255, 255),
            'left_center_area_lower': (255, 255, 150),
        }

        self.split_mini_court_areas()
        
    def initialize_with_frame(self, frame, court_keypoints, net_bbox=None):
        """
        Инициализация с первым кадром и детекцией сетки
        """
        width_frame = frame.shape[1]
        if net_bbox is not None:
            self.net_line = self.area_analyzer.detect_net_from_bbox(frame, net_bbox)
        else:
            net_line = self.net_detector.get_net_line(width_frame, court_keypoints)
            if net_line:
                self.net_line = net_line
                self.area_analyzer.net_line = self.net_line
        
        if self.net_line:
            # Разделяем зоны с учетом сетки
            self.area_analyzer.split_areas_with_net(court_keypoints, frame, width_frame)
            # Также разделяем зоны на мини-корте
            self.split_mini_court_areas()
        else:
            print(" Сетка не обнаружена, используем базовые зоны")
            self.area_analyzer.areas = self.area_analyzer.base_areas.copy()
        
        return self.net_line
    
    def split_mini_court_areas(self):
        """Разделяет зоны мини-корта сеткой"""
        self.mini_areas_with_zones = {
            'left_doubles_area_upper': [[self.drawing_key_points[0], self.drawing_key_points[1]],
                                        [self.drawing_key_points[8], self.drawing_key_points[9]],
                                        [self.drawing_key_points[30], self.drawing_key_points[31]],
                                        [self.drawing_key_points[28], self.drawing_key_points[29]]],

            'left_doubles_area_lower': [[self.drawing_key_points[10], self.drawing_key_points[11]],
                                        [self.drawing_key_points[6], self.drawing_key_points[7]],
                                        [self.drawing_key_points[28], self.drawing_key_points[29]],
                                        [self.drawing_key_points[30], self.drawing_key_points[31]]],

            'right_doubles_area_upper': [[self.drawing_key_points[12], self.drawing_key_points[13]],
                                        [self.drawing_key_points[2], self.drawing_key_points[3]],
                                        [self.drawing_key_points[36], self.drawing_key_points[37]],
                                        [self.drawing_key_points[34], self.drawing_key_points[35]]],

            'right_doubles_area_lower': [[self.drawing_key_points[14], self.drawing_key_points[15]],
                                        [self.drawing_key_points[4], self.drawing_key_points[5]],
                                        [self.drawing_key_points[36], self.drawing_key_points[37]],
                                        [self.drawing_key_points[34], self.drawing_key_points[35]]],

            'upper_baseline_area': [[self.drawing_key_points[8], self.drawing_key_points[9]],
                                        [self.drawing_key_points[12], self.drawing_key_points[13]],
                                        [self.drawing_key_points[18], self.drawing_key_points[19]],
                                        [self.drawing_key_points[16], self.drawing_key_points[17]]],

            'lower_baseline_area': [[self.drawing_key_points[14], self.drawing_key_points[15]],
                                        [self.drawing_key_points[22], self.drawing_key_points[23]],
                                        [self.drawing_key_points[20], self.drawing_key_points[21]],
                                        [self.drawing_key_points[10], self.drawing_key_points[11]]],

            'right_center_area_upper': [[self.drawing_key_points[16], self.drawing_key_points[17]],
                                        [self.drawing_key_points[24], self.drawing_key_points[25]],
                                        [self.drawing_key_points[32], self.drawing_key_points[33]],
                                        [self.drawing_key_points[30], self.drawing_key_points[31]]],

            'right_center_area_lower': [[self.drawing_key_points[20], self.drawing_key_points[21]],
                                        [self.drawing_key_points[26], self.drawing_key_points[27]],
                                        [self.drawing_key_points[32], self.drawing_key_points[33]],
                                        [self.drawing_key_points[30], self.drawing_key_points[31]]],

            'left_center_area_upper': [[self.drawing_key_points[24], self.drawing_key_points[25]],
                                        [self.drawing_key_points[18], self.drawing_key_points[19]],
                                        [self.drawing_key_points[34], self.drawing_key_points[35]],
                                        [self.drawing_key_points[32], self.drawing_key_points[33]]],

            'left_center_area_lower': [[self.drawing_key_points[26], self.drawing_key_points[27]],
                                        [self.drawing_key_points[22], self.drawing_key_points[23]],
                                        [self.drawing_key_points[34], self.drawing_key_points[35]],
                                        [self.drawing_key_points[32], self.drawing_key_points[33]]]
        }
    
    def draw_mini_court_with_zones(self, frame, player_positions=None, ball_positions=None):
        """Рисует мини-корт с разделенными зонами, игроками и мячом"""
        frame_out = frame.copy()
        frame_out = self.draw_background_rectangle(frame_out)
        
        for area_name, points in self.mini_areas_with_zones.items():
            color = self.zone_colors.get(area_name, (200, 200, 200))
            
            polygon_points = []
            for point in points:
                polygon_points.append((int(point[0]), int(point[1])))

            if len(polygon_points) >= 3:
                overlay = frame_out.copy()
                cv2.fillPoly(overlay, [np.array(polygon_points, dtype=np.int32)], color)
                cv2.addWeighted(overlay, 0.8, frame_out, 0.2, 0, frame_out)
                cv2.polylines(frame_out, [np.array(polygon_points, dtype=np.int32)], True, color, 1)
        
        frame_out = self.draw_court(frame_out)
        
        if player_positions is not None and isinstance(player_positions, dict):
            frame_out = self.draw_points_on_mini_court(frame_out, player_positions, 
                                                    color=(0, 255, 0), radius=7)
        
        if ball_positions is not None and isinstance(ball_positions, dict):
            frame_out = self.draw_points_on_mini_court(frame_out, ball_positions,
                                                    color=(0, 255, 255), radius=5)
        
        return frame_out
    
    def draw_mini_court_without_zones(self, frame, player_positions=None, ball_positions=None):
        """Рисует мини-корт БЕЗ разделения зонами, но с игроками и мячом"""
        frame_out = frame.copy()

        cv2.rectangle(frame_out, 
              (int(self.drawing_key_points[0]), int(self.drawing_key_points[1])),
              (int(self.drawing_key_points[4]), int(self.drawing_key_points[5])), 
              (93, 141, 212), -1)

        frame_out = self.draw_background_rectangle(frame_out)
        frame_out = self.draw_court(frame_out)
        
        if player_positions is not None and isinstance(player_positions, dict):
            frame_out = self.draw_points_on_mini_court(frame_out, player_positions, 
                                                    color=(0, 255, 0), radius=7)
        
        if ball_positions is not None and isinstance(ball_positions, dict):
            frame_out = self.draw_points_on_mini_court(frame_out, ball_positions,
                                                    color=(0, 255, 255), radius=5)
            
        return frame_out

class CourtVisualizerWithNet:
    """Визуализатор с отображением зон и сетки - цвета совпадают с мини-кортом"""
    
    def __init__(self, area_analyzer):
        self.area_analyzer = area_analyzer
        
        self.zone_colors = {
            'left_doubles_area_upper': (200, 255, 200),
            'left_doubles_area_lower': (255, 150, 150),
            'right_doubles_area_upper': (255, 200, 200),
            'right_doubles_area_lower': (150, 255, 150),
            'upper_baseline_area': (100, 100, 255),
            'lower_baseline_area': (100, 100, 255),
            'right_center_area_upper': (255, 255, 200),
            'right_center_area_lower': (150, 255, 255),
            'left_center_area_upper': (200, 255, 255),
            'left_center_area_lower': (255, 255, 150),
        }
        
    def draw_court_with_net(self, frame, court_keypoints, net_line=None):
        """Рисует корт с зонами и сеткой - цвета совпадают с мини-кортом"""
        frame_out = frame.copy()
        
        net_y = net_line[1] if net_line is not None else None
        
        if net_y is not None and hasattr(self.area_analyzer, 'get_polygon_coords'):
            try:
                polygon_list = self.area_analyzer.get_polygon_coords(court_keypoints, frame, frame.shape[1], net_y, debug=False)
                
                for poly_info in polygon_list:
                    coords = poly_info['coords']
                    area_name = poly_info['name']
                    
                    if len(coords) >= 8:  # Минимум 4 точки
                        color = self.zone_colors.get(area_name, (200, 200, 200))
                        
                        points = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = int(coords[i])
                                y = int(coords[i + 1])
                                points.append((x, y))
                        
                        if len(points) >= 3:
                            overlay = frame_out.copy()
                            cv2.fillPoly(overlay, [np.array(points)], color)
                            cv2.addWeighted(overlay, 0.6, frame_out, 0.4, 0, frame_out)
                            
                            cv2.polylines(frame_out, [np.array(points)], True, color, 2)
                            
            except Exception as e:
                print(f" Ошибка при рисовании разделенных полигонов: {e}")
        
        # Рисуем ключевые точки корта
        for i, point in enumerate(court_keypoints):
            if len(point) >= 2:
                x = float(point[0])
                y = float(point[1])
                
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(frame_out, (int(x), int(y)), 6, (255, 255, 255), -1)
                    cv2.circle(frame_out, (int(x), int(y)), 6, (0, 0, 0), 1)
        
        # Рисуем линию сетки
        if net_line is not None:
            from utils import make_parallel_line
            
            def get_parallel_net_line(court_keypoints, net_y, width):
                x_10, y_10 = court_keypoints[10]
                x_11, y_11 = court_keypoints[11]
                line_10_11 = [x_10, y_10, x_11, y_11]
                net_line = make_parallel_line(line_10_11, net_y, width)
                return net_line
            
            net_line_coords = get_parallel_net_line(court_keypoints, net_y, frame.shape[1])
            cv2.line(frame_out, 
                    (int(net_line_coords[0]), int(net_line_coords[1])),
                    (int(net_line_coords[2]), int(net_line_coords[3])), 
                    (0, 0, 255), 5)
        
        return frame_out