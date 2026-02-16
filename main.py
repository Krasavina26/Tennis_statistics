from collections import defaultdict
import cv2
import numpy as np
from trackers import BallTrackerNew, PlayerTrackerNew, CourtDetectorNew, NetDetector
import sys
import time
import constants
from mini_court import MiniCourtWithNetDivision
from statistics import create_fullscreen_mini_court_video, SpeedCalculator
from shot_detector import SimpleShotDetector

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Выводит текстовый прогресс-бар в консоль"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()

def print_simple_progress(iteration, total, prefix=''):
    """Упрощенный прогресс-бар (без графики)"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    sys.stdout.write(f'\r{prefix} Прогресс: {iteration}/{total} ({percent}%)')
    sys.stdout.flush()
    
    if iteration == total:
        print()

def estimate_time_remaining(start_time, current, total):
    """Оценивает оставшееся время"""
    elapsed = time.time() - start_time
    if current > 0:
        time_per_frame = elapsed / current
        remaining_frames = total - current
        remaining_time = time_per_frame * remaining_frames
        
        if remaining_time > 3600:
            return f"{remaining_time/3600:.1f}ч"
        elif remaining_time > 60:
            return f"{remaining_time/60:.1f}м"
        else:
            return f"{remaining_time:.0f}с"
    return "расчет..."


def main():
    input_video_path = "input_videos/test.mp4"
    output_video_path = "output_videos/output_test.avi"
    
    ball_tracker = BallTrackerNew(model_path="models/ball.pt")
    player_tracker = PlayerTrackerNew(model_path="models/player.pt")
    court_detector = CourtDetectorNew(
        model_path="models/court.pt", 
        use_refine_kps=False, 
        use_net_mask=True
    )
    net_model_path = "models/net.pt"
    net_detector = NetDetector(model_path=net_model_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f" Ошибка открытия видео: {input_video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f" Ошибка создания видеофайла")
        return
    
    
    frame_count = 0
    start_time = time.time()

    cap_first = cv2.VideoCapture(input_video_path)
    success, first_frame = cap_first.read()
    width_frame = first_frame.shape[1]
    cap_first.release()
    
    if not success:
        print(" Не удалось прочитать первый кадр!")
        return
    
    first_frame_court_kps = court_detector.detect_keypoints(first_frame)
    # first_frame_court_kps = court_detector.refine_keypoints(first_frame, first_frame_court_kps)
    
    mini_court = MiniCourtWithNetDivision(first_frame, net_model_path)
    
    cap = cv2.VideoCapture(input_video_path)
    all_ball_detections = []
    all_player_detections = []
    player_tracks = defaultdict(list)
    all_court_keypoints = []
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        best_ball = ball_tracker.get_best_detection(frame)
        all_ball_detections.append(best_ball)
        
        player_dets = player_tracker.get_player_detections(frame)
        all_player_detections.append(player_dets)
        
        all_court_keypoints.append(first_frame_court_kps.copy())
        
        frame_count += 1
        
        if frame_count % 10 == 0 or frame_count == total_frames:
            remaining_time = estimate_time_remaining(start_time, frame_count, total_frames)
            print_progress_bar(
                iteration=frame_count,
                total=total_frames,
                prefix=' Обработка кадров',
                suffix=f'Осталось: {remaining_time}',
                length=30
            )
    
    cap.release()
    
    positions = ball_tracker.get_positions_for_interpolation(all_ball_detections)
    interpolated = ball_tracker.interpolate_positions(positions, max_gap=5)
    
    filtered_players = player_tracker.choose_and_filter_players_each_frame(all_court_keypoints, all_player_detections)

    if first_frame_court_kps is not None:
        if isinstance(first_frame_court_kps, np.ndarray):
            court_kps_flat = first_frame_court_kps.flatten().tolist()
        else:
            court_kps_flat = first_frame_court_kps
        
        mini_court.initialize_homography_from_court_keypoints(court_kps_flat)

    player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        filtered_players,
        interpolated,
        court_kps_flat
    )

    frames_for_net = []
    cap_temp = cv2.VideoCapture(input_video_path)
    for _ in range(min(10, total_frames)):
        success, frame = cap_temp.read()
        if success:
            frames_for_net.append(frame)
    cap_temp.release()
    
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0
    start_time = time.time()
    
    speed_calculator = SpeedCalculator(fps=fps, court_width_meters=constants.DOUBLE_LINE_WIDTH, court_length_meters=constants.COURT_HEIGHT)
    
    player_speed_histories = defaultdict(list)

    extremums_y = ball_tracker.get_all_extremums(interpolated)
    candidate_frames_y = [e['frame'] for e in extremums_y]

    candidate_frames = candidate_frames_y
        
    simple_shot_detector = SimpleShotDetector(fps=fps, drawing_rectangle_width=mini_court.court_drawing_width, drawing_rectangle_height=mini_court.court_drawing_height)

    detected_events = simple_shot_detector.detect_events(
        candidate_frames=candidate_frames,
        ball_positions=ball_mini,
        player_positions=player_mini
    )

    shots_by_frame = simple_shot_detector.shots_by_frame
    stats = simple_shot_detector.get_statistics()
    simple_shot_detector.save_detailed_statistics()

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        annotated = frame.copy()
        
        if mini_court:
                player_pos = player_mini[frame_count] if frame_count < len(player_mini) else None
                ball_pos = ball_mini[frame_count] if frame_count < len(ball_mini) else None
                
                player_speeds = {}
                
                if ball_pos is not None:
                    ball_pos_for_draw  = ball_pos        
                
                if player_pos is not None:
                    for player_id, position in player_pos.items():
                        if position is not None:
                            player_bbox = None
                            if frame_count < len(filtered_players):
                                curr_players = filtered_players[frame_count]
                                if player_id in curr_players:
                                    player_bbox = curr_players[player_id]
                            
                            player_speed_m_s, player_speed_kmh = speed_calculator.update_player_speed(
                                player_id, position, player_bbox=player_bbox
                            )
                            
                            player_speed_histories[player_id].append(player_speed_kmh)
                            if len(player_speed_histories[player_id]) > 10:
                                player_speed_histories[player_id].pop(0)
                            
                            if player_speed_histories[player_id]:
                                player_speed_kmh = np.mean(player_speed_histories[player_id])
                            
                            player_speeds[player_id] = player_speed_kmh
                
                annotated = mini_court.draw_mini_court_without_zones(
                    annotated,
                    player_positions=player_pos,
                    ball_positions=ball_pos_for_draw
                )
                
                annotated = speed_calculator.draw_speed_info(
                    annotated, 
                    player_speeds,
                    position=(width - 350, height - 180)
                )

        out.write(annotated)
        frame_count += 1
        
        if frame_count % 10 == 0 or frame_count == total_frames:
            remaining_time = estimate_time_remaining(start_time, frame_count, total_frames)
            print_progress_bar(
                iteration=frame_count,
                total=total_frames,
                prefix=' Визуализация',
                suffix=f'Осталось: {remaining_time}',
                length=30
            )
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    fullscreen_video_path = "output_videos/fullscreen_mini_court_trajectories_test.avi"
    fullscreen_image_path = "output_videos/fullscreen_mini_court_summary_test.jpg"
    
    fullscreen_width = 1280
    fullscreen_height = 720
    
    result = create_fullscreen_mini_court_video(
        input_video_path=input_video_path,
        output_video_path=fullscreen_video_path,
        output_image_path=fullscreen_image_path,
        player_mini=player_mini,
        ball_mini=ball_mini,
        mini_court_data=mini_court,
        fps=fps,
        player_avg_heights=speed_calculator.player_avg_heights,
        trajectory_length=30,
        width=fullscreen_width,
        height=fullscreen_height
    )

    player_distances_meters = result['player_distances']

    with open('output_videos/game_statistics.txt', 'a', encoding='utf-8') as f:
        f.write("\nХАРАКТЕРИСТИКИ ИГРОКОВ\n")
        for player_id in [1, 2]:
            f.write(f"Игрок {player_id}:\n")
            if player_distances_meters[player_id] > 0:
                f.write(f"  Общая длина траектории: {player_distances_meters[player_id]:.2f} метров\n")
            if player_speed_histories[player_id]:
                    f.write(f"  Средняя скорость: {np.mean(player_speed_histories[player_id]):.2f} км/ч\n")
                    f.write(f"  Максимальная скорость: {max(player_speed_histories[player_id]):.2f} км/ч\n")


if __name__ == "__main__":
    main()