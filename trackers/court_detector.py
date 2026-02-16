import cv2
import numpy as np
from ultralytics import YOLO
import os
from utils import line_intersection
import matplotlib.pyplot as plt

class CourtDetectorNew:
    def __init__(self, model_path=r"C:\\Users\\krask\\Documents\\keypoint_detect\\runs\\pose\\runs\\yolo11m_pose\\court_yolo11m_pose_v510\\weights\\best.pt", use_refine_kps=True, use_net_mask=True):
        self.model = YOLO(model_path)
        self.model.conf = 0.3
        self.model.iou = 0.5
        self.use_refine_kps = use_refine_kps
        self.use_net_mask = use_net_mask
        self.net_model = YOLO(r"C:\\Users\\krask\\Documents\\tennis_analysis\\training\\player_detection\\runs\\padel_do_train\\weights\\best.pt") if use_net_mask else None
        
        self.track_id = None
        self.fixed_kps = np.full((14, 2), np.nan)
        self.debug_frame_count = 0
        self.keypoint_2_out = False
        self.keypoint_3_out = False

    def find_out_keypoints(self, frame, xy, ind):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gray = cv2.bilateralFilter(enhanced, 10, 30, 70)
        cv2.imwrite("find_gray.jpg", gray)

        if ind == 2:
                xy[ind] = [frame.shape[1], xy[7][1]]
                self.keypoint_2_out = True
                new_point = np.array([frame.shape[1], self.find_nearest_white_pixel(gray, frame.shape[1]-1, xy[7][1]-50)])
                xy = np.append(xy, [new_point], axis=0)

        elif ind == 3:
                xy[ind] = [0, xy[5][1]]
                self.keypoint_3_out = True
                new_point = np.array([0, self.find_nearest_white_pixel(gray, 0+1, xy[5][1]-50)])
                xy = np.append(xy, [new_point], axis=0)

        return xy
    
    def find_nearest_white_pixel(self, gray_image, x, start_y, threshold=110, max_search=100, debug=False):
        start_y = int(start_y)
        x_int = int(x)
        
        if debug:
            print(f"Поиск белого пикселя: x={x_int}, start_y={start_y}, threshold={threshold}")
        
        for distance in range(1, max_search + 1):
            y = start_y - distance
            
            if y < 0:
                if debug:
                    print(f"Достигнут верхний край изображения")
                break
            
            pixel_value = gray_image[y, x_int]
            
            if pixel_value >= threshold:
                if debug:
                    print(f"Найден белый пиксель: y={y}, значение={pixel_value}, расстояние={distance}")
                return y
            
            if distance <= 10:
                if debug:
                    print(f"  Проверка y={y}: значение={pixel_value}")
        
        if debug:
            print(f"Белый пиксель не найден в радиусе {max_search} пикселей")
        return None

    def detect_keypoints(self, frame):
        results = self.model.predict(
            frame,
            conf=0.3,
            iou=0.5,
            verbose=False
        )
        if not results or len(results) == 0 or results[0].keypoints is None:
            if not np.all(np.isnan(self.fixed_kps)):
                return np.copy(self.fixed_kps)
            else:
                return np.full((14, 2), np.nan)

        kpts = results[0].keypoints
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            court_idx = np.argmax(confs)
        else:
            if not np.all(np.isnan(self.fixed_kps)):
                return np.copy(self.fixed_kps)
            else:
                return np.full((14, 2), np.nan)

        xy = kpts.xy[court_idx].cpu().numpy()

        if kpts.has_visible:
            conf = kpts.conf[court_idx].cpu().numpy()
            xy[conf < 0.3] = np.nan
        
        if np.isnan(xy[2][0]):
            xy = self.find_out_keypoints(frame, xy, 2)

        if np.isnan(xy[3][0]):
            xy = self.find_out_keypoints(frame, xy, 3)

        if self.use_refine_kps:
            xy = self.refine_keypoints(frame, xy)

        return xy.astype(float)

    def refine_keypoints(self, frame, keypoints):
        refined = np.copy(keypoints)
        img_h, img_w = frame.shape[:2]
        
        debug_dir = "debug_crops"
        os.makedirs(debug_dir, exist_ok=True)

        net_bbox = None
        if self.use_net_mask and self.net_model:
            net_results = self.net_model(frame, verbose=False, classes=[2])
            if net_results and net_results[0].boxes is not None:
                boxes = net_results[0].boxes.xyxy.cpu().numpy()
                confs = net_results[0].boxes.conf.cpu().numpy()
                if len(boxes) > 0:
                    max_conf_idx = np.argmax(confs)
                    net_bbox = boxes[max_conf_idx].astype(int)

        for i in range(keypoints.shape[0]):
            x, y = keypoints[i]
            if np.isnan(x) or np.isnan(y):
                continue

            crop_size = 40
            y_min = max(0, int(y - crop_size))
            y_max = min(img_h, int(y + crop_size))
            x_min = max(0, int(x - crop_size))
            x_max = min(img_w, int(x + crop_size))

            if y_max <= y_min or x_max <= x_min or (y_max - y_min) < 40 or (x_max - x_min) < 40:
                continue

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            point_dir = os.path.join(debug_dir, f"kp_{i:02d}")
            os.makedirs(point_dir, exist_ok=True)

            debug_info = []
            debug_info.append(f"=== Обработка Keypoint {i} ===")
            debug_info.append(f"Исходные координаты: x={x:.1f}, y={y:.1f}")
            debug_info.append(f"Crop region: y={y_min}:{y_max}, x={x_min}:{x_max}")
            debug_info.append(f"Crop размер: {crop.shape[1]}x{crop.shape[0]}")
            
            point = None

            # Обработка изображения
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(gray, 10, 30, 70)
            _, white_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2, 2), np.uint8)
            white_mask_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            edges_d = cv2.dilate(white_mask_clean, kernel, iterations=1)
            edges = cv2.erode(edges_d, kernel, iterations=2)
            edges_without_net = edges.copy()

            # Применение маски сети
            if net_bbox is not None:
                net_x1, net_y1, net_x2, net_y2 = net_bbox
                crop_net_y1 = max(0, net_y1 - y_min)
                crop_net_x1 = max(0, net_x1 - x_min)
                crop_net_y2 = min(edges.shape[0], net_y2 - y_min)
                crop_net_x2 = min(edges.shape[1], net_x2 - x_min)
                
                if crop_net_y1 < crop_net_y2 and crop_net_x1 < crop_net_x2:
                    edges_without_net[crop_net_y1:crop_net_y2, crop_net_x1:crop_net_x2] = 0
                    debug_info.append(f"Применена маска сети: y={crop_net_y1}:{crop_net_y2}, x={crop_net_x1}:{crop_net_x2}")

            # Детекция линий
            lines = self.detect_lines(edges_without_net)
            
            if lines is not None and len(lines) > 0:
                debug_info.append(f"Hайдено {len(lines)} линий")

                if len(lines) < 2:
                    debug_info.append(f"ПРОПУСК: слишком мало линий (<2)")
                else:
                    selected_indices = self.filter_lines(lines)
                    debug_info.append(f"После фильтрации {len(selected_indices)} линий")

                    if len(selected_indices) >= 2:
                        line1, line2 = lines[selected_indices[0]], lines[selected_indices[1]]
                        try:
                            point = line_intersection(line1[0], line2[0])
                            if point:
                                debug_info.append(f"Найдена точка пересечения {point}")
                                x_new, y_new = point
                                # Конвертируем координаты из crop в глобальные
                                x_new_global = x_min + x_new
                                y_new_global = y_min + y_new
                                refined[i] = [x_new_global, y_new_global]
                            else:
                                debug_info.append(f"Линии не пересекаются в пределах отрезков")
                        except Exception as e:
                            debug_info.append(f"Ошибка в line_intersection: {e}")
                    else:
                        debug_info.append(f"После фильтрации недостаточно линий")
            else:
                debug_info.append(f"Не найдено линий")

            # Визуализация (используем matplotlib)
            try:
                fig, axs = plt.subplots(2, 4, figsize=(10, 5))
                axs = axs.flatten()
                
                axs[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                axs[0].set_title('Origin')
                axs[0].axis('off')

                axs[1].imshow(gray, cmap='gray')
                axs[1].set_title('Grayscale')
                axs[1].axis('off')

                axs[2].imshow(blurred, cmap='gray')
                axs[2].set_title('Blurred')
                axs[2].axis('off')

                axs[3].imshow(white_mask_clean, cmap='gray')
                axs[3].set_title('Canny Edges')
                axs[3].axis('off')

                axs[4].imshow(edges_d, cmap='gray')
                axs[4].set_title('Dilation')
                axs[4].axis('off')

                axs[5].imshow(edges, cmap='gray')
                axs[5].set_title('Erosion')
                axs[5].axis('off')

                axs[6].imshow(edges_without_net, cmap='gray')
                axs[6].set_title('Net Mask')
                axs[6].axis('off')
                
                axs[7].imshow(edges_without_net, cmap='gray')
                
                if lines is not None and len(lines) >= 2 and 'selected_indices' in locals() and len(selected_indices) >= 2:
                    for idx in selected_indices[:2]:  # Только первые две линии
                        x1, y1, x2, y2 = lines[idx][0]
                        color = 'r' if idx == selected_indices[0] else 'b'
                        axs[7].plot([x1, x2], [y1, y2], color, linewidth=2)
                
                axs[7].set_title('Lines Intersection')
                axs[7].axis('off')
                
                if point:
                    axs[7].plot(point[0], point[1], 'go', markersize=10)
                
                plt.tight_layout()
                plt.savefig(os.path.join(point_dir, "refining_epoches.jpg"), dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                debug_info.append(f"Ошибка при сохранении графика: {e}")

            # Сохраняем отладочную информацию в файл
            with open(os.path.join(point_dir, "debug_info.txt"), "w") as f:
                f.write("\n".join(debug_info))

            # Сохраняем краткую информацию в info.txt
            with open(os.path.join(point_dir, "info.txt"), "w") as f:
                f.write(f"Keypoint {i}\n")
                f.write(f"Исходные координаты: x={x:.1f}, y={y:.1f}\n")
                if point:
                    f.write(f"Точка пересечения: ({point[0]:.1f}, {point[1]:.1f})\n")
                    f.write(f"Глобальные координаты: ({x_min + point[0]:.1f}, {y_min + point[1]:.1f})\n")
                    f.write(f"Успешно уточнена: ДА\n")
                else:
                    f.write(f"Точка пересечения: НЕ НАЙДЕНА\n")

        self.debug_frame_count += 1
        return refined

    def detect_lines(self, img):
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(img)[0]
        return lines
    
    def filter_lines(self, lines):
        if lines is None or len(lines) < 2:
            print("Не удалось найти достаточно линий.")
            return []
        
        lines_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 1e-6:
                theta = np.arctan2(y2 - y1, x2 - x1)
                if theta < 0:
                    theta += np.pi
                rho = x1 * np.cos(theta) + y1 * np.sin(theta)
                if rho < 0:
                    rho = -rho
                    theta -= np.pi
                if theta < 0:
                    theta += np.pi
            else:
                theta = np.pi/2
                rho = x1
            
            lines_params.append([rho, theta])
        
        # Выбор по длине
        line_lengths = [np.linalg.norm(lines[i][0][:2] - lines[i][0][2:]) for i in range(len(lines))]
        sorted_indices = sorted(range(len(line_lengths)), key=lambda i: -line_lengths[i])
        
        selected_params = []
        selected_indices = []
        
        # Первая линия
        selected_indices.append(sorted_indices[0])
        selected_params.append(lines_params[sorted_indices[0]])
        theta1 = selected_params[0][1]
        
        # Вторая линия с отличающимся углом
        for idx in sorted_indices[1:]:
            theta2 = lines_params[idx][1]
            if abs(theta1 - theta2) > np.pi / 18 and abs(theta1 - theta2) < np.pi - np.pi/18:
                selected_indices.append(idx)
                selected_params.append(lines_params[idx])
                break
        
        if len(selected_params) < 2:
            selected_indices = sorted_indices[:2]

        return selected_indices[:2]

    def draw_keypoints_on_video(self, frame, kps):
        frame_out = frame.copy()
        for i, (x, y) in enumerate(kps):
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame_out, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.putText(frame_out, str(i), (int(x)+10, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame_out