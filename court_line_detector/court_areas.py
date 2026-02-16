import numpy as np
from utils import line_intersection, make_parallel_line, point_inside_polygon

class CourtAreaAnalyzerWithNet:
    """Анализатор зон с разделением сеткой"""
    
    def __init__(self, net_detector_path=None):
        self.base_areas = {
            'left_doubles_area': [0, 4, 5, 3],  # зона 1 + 6
            'right_doubles_area': [6, 1, 2, 7], # зона 3 + 4
            'upper_baseline_area': [4, 8, 9, 6],   # зона 2
            'lower_baseline_area': [5, 10, 11, 7], # зона 5
            'right_center_area': [8, 12, 13, 10], # зона 8 + 7
            'left_center_area': [12, 9, 11, 13] # зона 8 + 7
        }


        self.areas = self.base_areas.copy()
        
        self.net_line = None  # [x1, y1, x2, y2] - линия сетки
        self.net_detector = None
    
    def detect_net_from_bbox(self, frame, net_bbox):
        """Определяет линию сетки из bounding box сетки"""
        if net_bbox is not None and len(net_bbox) == 4:
            x1, y1, x2, y2 = net_bbox
            
            self.net_line = [0, y2, frame.shape[1], y2]
            return self.net_line
        
        return None

    def get_polygon_coords(self, court_keypoints, frame, width, net_y=None, debug=False):
        """Получает координаты всех полигонов с разделением сеткой"""
        polygon_list = []
        if court_keypoints.shape[0] == 15:
            self.base_areas = {
            'left_doubles_area': [0, 4, 5, 3],  # зона 1 + 6
            'right_doubles_area': [6, 1, 14, 2, 7], # зона 3 + 4
            'upper_baseline_area': [4, 8, 9, 6],   # зона 2
            'lower_baseline_area': [5, 10, 11, 7], # зона 5
            'right_center_area': [8, 12, 13, 10], # зона 8 + 7
            'left_center_area': [12, 9, 11, 13] # зона 8 + 7
        }
        
        if court_keypoints.shape[0] == 16:
            self.base_areas = {
            'left_doubles_area': [0, 4, 5, 3, 15],  # зона 1 + 6
            'right_doubles_area': [6, 1, 14, 2, 7], # зона 3 + 4
            'upper_baseline_area': [4, 8, 9, 6],   # зона 2
            'lower_baseline_area': [5, 10, 11, 7], # зона 5
            'right_center_area': [8, 12, 13, 10], # зона 8 + 7
            'left_center_area': [12, 9, 11, 13] # зона 8 + 7
        }
        
        def get_points_for_indices(indices):
            """Получаем точки по индексам"""
            points = []
            for idx in indices:
                if idx < court_keypoints.shape[0]:
                    x = court_keypoints[idx, 0]
                    y = court_keypoints[idx, 1]
                    if np.isscalar(x) and np.isscalar(y):
                        if not np.isnan(x) and not np.isnan(y):
                            points.append((float(x), float(y), idx))
            return points
        
        def split_area_by_point(intersect_points, points):
            """Разделяет полигон по точке пересечения"""
            
            if isinstance(intersect_points[0], (list, tuple)):
                y_intersect = float(intersect_points[0][1])
            else:
                y_intersect = float(intersect_points[0])
            
            converted_intersections = []
            for intersect in intersect_points:
                if isinstance(intersect, (list, tuple)) and len(intersect) >= 2:
                    x_intersect = float(intersect[0])
                    y_intersect = float(intersect[1])
                    converted_intersections.append([x_intersect, y_intersect])
               
            converted_intersections.sort(key=lambda pt: pt[0])
            
            upper_ordered = []
            lower_ordered = []
            
            for point in points:
                x, y, idx = point
                if y < y_intersect:
                    upper_ordered.append([x, y])
                elif y > y_intersect:
                    lower_ordered.append([x, y])
            
            if len(converted_intersections) >= 2:
                x1, y1 = converted_intersections[-1]
                x2, y2 = converted_intersections[0]
                upper_ordered.append([x1, y1])
                upper_ordered.append([x2, y2])
                lower_ordered.append([x2, y2])
                lower_ordered.append([x1, y2])
            
            upper_coords = [coord for pt in upper_ordered for coord in pt]
            lower_coords = [coord for pt in lower_ordered for coord in pt]
            
            return upper_coords, lower_coords
        
        # ----------------------------------------------
        # Обрабатываем каждую базовую зону
        for area_name, indices in self.base_areas.items():
            if 'left' in area_name or 'right' in area_name:
                points = get_points_for_indices(indices)
                
                if debug:
                    print(f"\n Обработка зоны '{area_name}' с индексами {indices}")
                    print(f"   Найдено точек: {len(points)}")
                    if points:
                        print(f"   Координаты: {[(x, y) for x, y, _ in points]}")
                
                if net_y is not None:
                    if debug:
                        print(f"   Пытаемся разделить сеткой на Y={net_y}")
                    
                    pts = []
                    for x, y, idx in points:
                        pts.append([x, y])

                    if debug:
                        print(f"\n Pts '{pts}")  

                    line_to_intersect = []
                    if len(pts) == 4:
                        line_to_intersect.append(pts[1] + pts[2])
                        line_to_intersect.append(pts[3] + pts[0])
                    if len(pts) == 5:
                        line_to_intersect.append(pts[1] + pts[2])
                        line_to_intersect.append(pts[4] + pts[0])

                    if debug:
                        print(f"\n Линии для пересечения: '{line_to_intersect}")
                    
                    def get_parallel_net_line(court_keypoints):
                        if court_keypoints.shape[0] > 11:
                            x_10, y_10 = court_keypoints[10]
                            x_11, y_11 = court_keypoints[11]
                            line_10_11 = [x_10, y_10, x_11, y_11]
                            return make_parallel_line(line_10_11, net_y, width)
                        return [0, net_y, width, net_y]
                    
                    self.net_line = get_parallel_net_line(court_keypoints)
                    
                    intersect_points = []
                    for i in range(len(line_to_intersect)):
                        if isinstance(line_to_intersect[i], list) and len(line_to_intersect[i]) == 4:
                            intersection = line_intersection(line_to_intersect[i], self.net_line)
                            if intersection:
                                intersect_points.append(intersection)
                    
                    if debug:
                        print(f"   Точки пересечения: {intersect_points}")
                        
                    upper_poly, lower_poly = split_area_by_point(intersect_points, points)
                    if debug:
                            print(f"    Созданы полигоны {upper_poly, lower_poly}")
                    
                    if upper_poly and len(upper_poly) >= 8:  # Минимум 4 точки (8 координат)
                        if debug:
                            print(f"    Создан верхний полигон ({len(upper_poly)//2} точек)")
                        
                        polygon_list.append({
                            'name': f"{area_name}_upper",
                            'coords': upper_poly,
                            'indices': indices,
                            'net_y': net_y
                        })
                    
                    if lower_poly and len(lower_poly) >= 8:  # Минимум 4 точки (8 координат)
                        if debug:
                            print(f"    Создан нижний полигон ({len(lower_poly)//2} точек)")
                        
                        polygon_list.append({
                            'name': f"{area_name}_lower",
                            'coords': lower_poly,
                            'indices': indices,
                            'net_y': net_y
                        })
                    
                    if not upper_poly and not lower_poly:
                        if debug:
                            print(f"    Не удалось разделить зону")
                        
                        # Если не удалось разделить, используем исходный полигон
                        if len(points) >= 4:
                            coords = []
                            for x, y, idx in points:
                                coords.extend([x, y])
                            
                            polygon_list.append({
                                'name': area_name,
                                'coords': coords,
                                'indices': indices,
                                'net_y': None
                            })
                else:
                    # Не разделяем, просто создаем полигон
                    if len(points) >= 4:
                        coords = []
                        for x, y, idx in points:
                            coords.extend([x, y])
                        
                        if debug:
                            print(f"    Создан неразделенный полигон ({len(coords)//2} точек)")
                        
                        polygon_list.append({
                            'name': area_name,
                            'coords': coords,
                            'indices': indices,
                            'net_y': None
                        })
                    elif debug:
                        print(f"    Недостаточно точек для полигона")
            
            else:
                # Для других зон (не left/right) просто добавляем их без разделения
                points = get_points_for_indices(indices)
                if len(points) >= 4:
                    coords = []
                    for x, y, idx in points:
                        coords.extend([x, y])
                    
                    polygon_list.append({
                        'name': area_name,
                        'coords': coords,
                        'indices': indices,
                        'net_y': None
                    })
        
        if debug:
            print(f"\n Итог: создано {len(polygon_list)} полигонов")
            for poly in polygon_list:
                print(f"   - {poly['name']}: {len(poly['coords'])//2} точек")
        
        return polygon_list
    

    def get_area_for_point(self, point, court_keypoints, debug=False):
        """Находит зону для точки с учетом разделения сеткой"""
        if self.net_line is None:
            self.areas = self.base_areas.copy()
        
        for area_name, point_indices in self.areas.items():
            polygon_points = []
            for idx in point_indices:
                if idx < len(court_keypoints) and len(court_keypoints[idx]) >= 2:
                    x = court_keypoints[idx][0]
                    y = court_keypoints[idx][1]
                    if not np.isnan(x) and not np.isnan(y):
                        polygon_points.append((float(x), float(y)))
            
            if len(polygon_points) >= 3:
                if point_inside_polygon(point[0], point[1], polygon_points):
                    if debug:
                        print(f'Точка: {point}, Зона: {area_name}, Полигон: {polygon_points}')
                    return area_name, point_indices
        return "outside", None
        
    def split_areas_with_net(self, court_keypoints, frame, width, debug=False):
        """Разделяет зоны на верхние и нижние части с помощью линии сетки"""
        if self.net_line is None:
            self.areas = self.base_areas.copy()
            if debug:
                print(" Сетка не определена, используем базовые зоны")
            return
        
        net_y = self.net_line[1]
        polygon_list = self.get_polygon_coords(court_keypoints, frame, width, net_y)
        
        self.areas = {}
        for poly_info in polygon_list:
            self.areas[poly_info['name']] = poly_info['indices']
        
        if debug:
            print(f" Создано {len(self.areas)} зон с разделением сеткой")
            for area_name in self.areas.keys():
                print(f"   - {area_name}")