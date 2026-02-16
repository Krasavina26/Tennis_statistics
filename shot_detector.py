from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class ShotEvent:
    """Событие удара или подачи"""
    frame: int
    player_id: int
    event_type: str  # 'serve', 'shot', 'bounce', 'error'
    ball_position: Tuple[float, float]
    confidence: float
    timestamp: float
    rally_id: Optional[int] = None
    speed_kmh: Optional[float] = None
    
    def __post_init__(self):
        if self.event_type == 'serve':
            self.is_serve = True
        else:
            self.is_serve = False

@dataclass
class ServeStats:
    """Статистика подач для игрока"""
    first_serves: int = 0
    first_serves_in: int = 0
    second_serves: int = 0
    double_faults: int = 0
    total_serves: int = 0
    successful_serves: int = 0
    
    @property
    def first_serve_percentage(self) -> float:
        """Доля успешных первых подач"""
        if self.first_serves == 0:
            return 0.0
        return (self.first_serves_in / self.first_serves) * 100
    
    @property
    def double_fault_percentage(self) -> float:
        """Доля двойных ошибок"""
        if self.total_serves == 0:
            return 0.0
        return (self.double_faults / self.total_serves) * 100
    
    @property
    def serve_success_percentage(self) -> float:
        """Общая успешность подач"""
        if self.total_serves == 0:
            return 0.0
        return (self.successful_serves / self.total_serves) * 100
    
    def __str__(self) -> str:
        return (f"Подачи: {self.total_serves} всего, {self.successful_serves} успешных ({self.serve_success_percentage:.1f}%)\n"
                f"Первая подача: {self.first_serves_in}/{self.first_serves} ({self.first_serve_percentage:.1f}%)\n"
                f"Вторая подача: {self.second_serves} всего\n"
                f"Двойные ошибки: {self.double_faults} ({self.double_fault_percentage:.1f}%)")

@dataclass
class Rally:
    """Розыгрыш (ралли)"""
    rally_id: int
    start_frame: int
    end_frame: Optional[int] = None
    shots: List[ShotEvent] = None
    serving_player: Optional[int] = None
    receiving_player: Optional[int] = None
    winner: Optional[int] = None
    error_type: Optional[str] = None
    serve_success: Optional[bool] = None
    is_double_fault: bool = False
    
    def __post_init__(self):
        if self.shots is None:
            self.shots = []
    
    @property
    def duration_frames(self):
        if self.end_frame:
            return self.end_frame - self.start_frame
        return 0
    
    @property
    def shot_count(self):
        return len(self.shots)
    
    @property
    def is_completed(self):
        return self.end_frame is not None
    
    @property
    def serve_frame(self):
        """Возвращает кадр подачи (если есть)"""
        for shot in self.shots:
            if shot.event_type == 'serve':
                return shot.frame
        return None
    
    @property
    def serve_events(self) -> List[ShotEvent]:
        """Возвращает все события подач в розыгрыше"""
        return [shot for shot in self.shots if shot.event_type == 'serve']
    
    @property
    def first_serve_event(self) -> Optional[ShotEvent]:
        """Возвращает первую подачу в розыгрыше (если есть)"""
        for shot in self.shots:
            if shot.event_type == 'serve':
                return shot
        return None

class SimpleShotDetector:
    """Упрощенный детектор ударов/подач на мини-корте"""
    
    def __init__(self, fps, drawing_rectangle_width, drawing_rectangle_height):
        self.fps = fps
        self.COURT_WIDTH = drawing_rectangle_width
        self.COURT_HEIGHT = drawing_rectangle_height
        self.NET_Y = self.COURT_HEIGHT // 2
        
        self.PROXIMITY_THRESHOLD = 70
        self.MIN_SHOT_INTERVAL = 8
        self.SERVE_MAX_FRAME = fps * 5
        self.RALLY_TIMEOUT = fps * 3
        self.SAME_PLAYER_WINDOW = 15
        self.CONSECUTIVE_WINDOW = 10
        
        self.shots: List[ShotEvent] = []
        self.rallies: List[Rally] = []
        self.current_rally: Optional[Rally] = None
        
        self.game_state = {
            'serve_detected': False,
            'current_server': None,
            'last_shot_frame': -1000,
            'last_shot_player': None,
            'consecutive_shots_same_player': 0,
            'last_event_type': None,
            'first_shot_in_game': True,
            'serve_count_in_rally': 0,
        }
        
        self.shots_by_frame = {}
        
        self.serve_stats = {
            1: ServeStats(),
            2: ServeStats()
        }
        
        self.stats = {
            'total_events': 0,
            'serves': 0,
            'shots': 0,
            'bounces': 0,
            'player_stats': {
                1: {'shots': 0, 'serves': 0},
                2: {'shots': 0, 'serves': 0},
            }
        }
        
        self.debug_info = []

    def get_shots_with_speeds(self):
        """Возвращает список ударов с их скоростями"""
        shots_with_speeds = []
        for event in self.shots:
            if hasattr(event, 'speed_kmh') and event.speed_kmh is not None:
                shots_with_speeds.append({
                    'frame': event.frame,
                    'player_id': event.player_id,
                    'event_type': event.event_type,
                    'speed_kmh': event.speed_kmh
                })
        return shots_with_speeds
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Расстояние между двумя точками"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_closest_player_in_frame(self, frame_idx: int, ball_positions: List, 
                                    player_positions: List[Dict[int, any]]) -> Tuple[Optional[int], float, Optional[Tuple]]:
        """
        Находит ближайшего игрока в указанном кадре.
        Возвращает (player_id, distance, player_position) или (None, inf, None).
        """
        ball_pos = self.get_ball_position(ball_positions, frame_idx)
        if ball_pos is None:
            return None, float('inf'), None
        
        players_in_frame = player_positions[frame_idx] if frame_idx < len(player_positions) else None
        if not isinstance(players_in_frame, dict):
            return None, float('inf'), None
        
        closest_player = None
        min_distance = float('inf')
        closest_player_pos = None
        
        for pid in players_in_frame:
            player_pos = self.get_player_position(player_positions, frame_idx, pid)
            if player_pos:
                distance = self.calculate_distance(ball_pos, player_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_player = pid
                    closest_player_pos = player_pos
        
        return closest_player, min_distance, closest_player_pos
    
    def find_best_consecutive_candidate(self, start_idx: int, candidate_frames: List[int], 
                                       ball_positions: List, player_positions: List[Dict[int, any]]) -> Tuple[int, int, float]:
        """
        Находит лучший кадр среди последовательных кандидатов с тем же ближайшим игроком.
        Возвращает (best_frame_idx, player_id, min_distance).
        """
        if start_idx >= len(candidate_frames):
            return -1, -1, float('inf')
        
        first_frame = candidate_frames[start_idx]
        first_player, first_distance, _ = self.find_closest_player_in_frame(
            first_frame, ball_positions, player_positions
        )
        
        if first_player is None:
            return first_frame, -1, float('inf')
        
        best_frame_idx = start_idx
        best_frame = first_frame
        best_distance = first_distance
        target_player = first_player
        
        # Ищем следующие кадры с тем же ближайшим игроком
        i = start_idx + 1
        while i < len(candidate_frames):
            current_frame = candidate_frames[i]
            current_player, current_distance, _ = self.find_closest_player_in_frame(
                current_frame, ball_positions, player_positions
            )
            
            if current_player == target_player:
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_frame_idx = i
                    best_frame = current_frame
                i += 1
            else:
                break
        
        return best_frame_idx, target_player, best_distance
    
    def detect_event_type(self, frame_idx: int, ball_pos: Tuple[float, float], 
                         player_pos: Tuple[float, float], player_id: int,
                         candidate_frames: List[int], ball_positions: List, 
                         distance_to_player: float, player_positions: List[Dict[int, any]]) -> Tuple[str, str]:
        """Определяет тип события (serve/shot/bounce) с отладочной информацией"""
        
        closest_distance = distance_to_player
        
        # Проверка на отскок (не рядом с игроком)
        if closest_distance > self.PROXIMITY_THRESHOLD:
            reason = f"Отскок: мяч далеко от игроков (расстояние: {closest_distance:.1f}px > порог {self.PROXIMITY_THRESHOLD}px)"
            
            if self.game_state['last_event_type'] == 'bounce':
                self.game_state['consecutive_bounces'] = self.game_state.get('consecutive_bounces', 0) + 1
            else:
                self.game_state['consecutive_bounces'] = 1
            
            if self.game_state.get('consecutive_bounces', 0) >= 3:
                self.game_state['after_many_bounces'] = True
            
            return 'bounce', reason
        
        self.game_state['consecutive_bounces'] = 0
        
        # Проверка на подачу: самый первый удар в игре считается подачей
        if self.game_state['first_shot_in_game']:
            reason = f"Подача: первый удар в игре (расстояние до игрока {player_id}: {closest_distance:.1f}px)"
            return 'serve', reason
        
        # Проверка на подачу: первый удар в розыгрыше (если есть активный розыгрыш)
        if self.current_rally and len(self.current_rally.shots) == 0:
            reason = f"Подача: первый удар в розыгрыше #{self.current_rally.rally_id} (расстояние до игрока {player_id}: {closest_distance:.1f}px)"
            return 'serve', reason
        
        # Проверка на подачу: если было много отскоков подряд, а теперь мяч близко к игроку
        if self.game_state.get('after_many_bounces', False) and not self.current_rally:
            reason = f"Подача: после серии отскоков мяч снова рядом с игроком {player_id} (расстояние: {closest_distance:.1f}px)"
            self.game_state['after_many_bounces'] = False  # Сбрасываем флаг
            return 'serve', reason
        
        # Проверка на обычный удар: если розыгрыш уже начался
        if self.current_rally:
            if self.game_state['last_shot_player'] is None or self.game_state['last_shot_player'] != player_id:
                reason = f"Удар: смена игрока (предыдущий: игрок {self.game_state['last_shot_player']}, текущий: игрок {player_id}, расстояние: {closest_distance:.1f}px)"
                return 'shot', reason
            else:
                reason = f"Удар: тот же игрок {player_id} (расстояние: {closest_distance:.1f}px)"
                return 'shot', reason
        
        # Если нет активного розыгрыша, но это не первый удар в игре
        if not self.current_rally:
            reason = f"Неизвестно: нет активного розыгрыша, но это не первый удар в игре (расстояние: {closest_distance:.1f}px)"
            return 'unknown', reason
        
        # Все остальные случаи
        reason = f"Неизвестно: непредвиденная ситуация (расстояние: {closest_distance:.1f}px)"
        return 'unknown', reason
    
    def get_ball_position(self, ball_positions: List, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Получает позицию мяча для кадра"""
        if frame_idx >= len(ball_positions):
            return None
        
        ball_data = ball_positions[frame_idx]
        if ball_data is None:
            return None
        
        if isinstance(ball_data, dict) and 1 in ball_data:
            pos = ball_data[1]
        else:
            pos = ball_data
        
        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
            return (float(pos[0]), float(pos[1]))
        
        return None
    
    def get_player_position(self, player_positions: List[Dict], frame_idx: int, 
                           player_id: int) -> Optional[Tuple[float, float]]:
        """Получает позицию игрока для кадра"""
        if frame_idx >= len(player_positions):
            return None
        
        players_in_frame = player_positions[frame_idx]
        if not isinstance(players_in_frame, dict):
            return None
        
        player_data = players_in_frame.get(player_id)
        if player_data is None:
            return None
        
        if isinstance(player_data, (list, tuple, np.ndarray)) and len(player_data) >= 2:
            return (float(player_data[0]), float(player_data[1]))
        
        return None
    
    def is_same_player_consecutive_event(self, frame_idx: int, player_id: int) -> Tuple[bool, str]:
        """Проверяет, является ли это событие слишком близким к предыдущему от того же игрока"""
        last_frame = self.game_state['last_shot_frame']
        last_player = self.game_state['last_shot_player']
        
        if last_player == player_id and last_frame > 0:
            frame_diff = frame_idx - last_frame
            return True, f"Объединение: игрок {player_id} уже ударил {frame_diff} кадров назад (кадр {last_frame})"
        
        return False, "OK"
    
    def start_new_rally(self, frame_idx: int, serve_event: Optional[ShotEvent] = None, debug=False):
        """Начинает новый розыгрыш"""
        rally_id = len(self.rallies) + 1
        
        self.current_rally = Rally(
            rally_id=rally_id,
            start_frame=frame_idx,
            serving_player=serve_event.player_id if serve_event else None,
            receiving_player=3 - serve_event.player_id if serve_event else None
        )
        
        if serve_event:
            serve_event.rally_id = rally_id
            self.current_rally.shots.append(serve_event)
        
        self.rallies.append(self.current_rally)
        self.game_state['consecutive_shots_same_player'] = 0
        self.game_state['serve_count_in_rally'] = 0
        
        if serve_event:
            self.game_state['first_shot_in_game'] = False
            self.game_state['serve_count_in_rally'] += 1
        
        if debug:
            print(f" Начат розыгрыш #{rally_id} на кадре {frame_idx}")
        if serve_event and debug:
                print(f"   Подающий: Игрок {serve_event.player_id}")
    
    def end_current_rally(self, frame_idx: int, winner: Optional[int] = None, 
                         error_type: Optional[str] = None, debug=False):
        """Завершает текущий розыгрыш"""
        if self.current_rally and not self.current_rally.is_completed:
            self.current_rally.end_frame = frame_idx
            self.current_rally.winner = winner
            self.current_rally.error_type = error_type
            
            # Определяем успешность подачи
            serve_events = self.current_rally.serve_events
            if serve_events:
                if len(self.current_rally.shots) > 1:
                    self.current_rally.serve_success = True
                elif error_type and winner != self.current_rally.serving_player:
                    self.current_rally.serve_success = False
                if self.current_rally.is_double_fault:
                    self.current_rally.serve_success = False
            if debug:        
                print(f" Розыгрыш #{self.current_rally.rally_id} завершен:")
                print(f"   Кадры: {self.current_rally.start_frame}-{frame_idx}")
                print(f"   Длительность: {self.current_rally.duration_frames} кадров")
                print(f"   Ударов: {self.current_rally.shot_count}")
                if winner:
                    print(f"   Победитель: Игрок {winner}")
                if error_type:
                    print(f"   Ошибка: {error_type}")
                if self.current_rally.serve_success is not None:
                    status = "успешная" if self.current_rally.serve_success else "неуспешная"
                    print(f"   Подача: {status}")
                if self.current_rally.is_double_fault:
                    print(f"     ДВОЙНАЯ ОШИБКА!")
            
            self.current_rally = None
            self.game_state['serve_detected'] = False
            self.game_state['current_server'] = None
            self.game_state['consecutive_shots_same_player'] = 0
            self.game_state['serve_count_in_rally'] = 0
    
    def update_serve_statistics(self, rally: Rally):
        """Обновляет статистику подач на основе завершенного розыгрыша"""
        if not rally.is_completed or not rally.serving_player:
            return
        
        player_id = rally.serving_player
        serve_stats = self.serve_stats[player_id]
        
        serve_events = rally.serve_events
        if not serve_events:
            return
        
        serve_stats.total_serves += len(serve_events)
        
        # Первая подача
        if len(serve_events) >= 1:
            serve_stats.first_serves += 1
            if rally.serve_success:
                serve_stats.first_serves_in += 1
                serve_stats.successful_serves += 1
        
        # Вторая подача
        if len(serve_events) >= 2:
            serve_stats.second_serves += 1
            if len(rally.shots) > 2 or (rally.winner and rally.winner == player_id):
                serve_stats.successful_serves += 1
        
        # Двойная ошибка
        if rally.is_double_fault:
            serve_stats.double_faults += 1
    
    def detect_events(self, candidate_frames: List[int], 
                     ball_positions: List, 
                     player_positions: List[Dict[int, any]],
                     debug=False) -> List[ShotEvent]:
        """Основная функция детекции событий с подробной отладкой"""
        if debug:
            print(f"\n Детекция событий на мини-корте...")
            print(f"   Всего кандидатов: {len(candidate_frames)}")
            print(f"   Порог близости: {self.PROXIMITY_THRESHOLD}px")
        
        detected_events = []
        
        i = 0
        while i < len(candidate_frames):
            frame_idx = candidate_frames[i]
            
            if frame_idx >= len(ball_positions) or frame_idx >= len(player_positions):
                if debug:
                    print(f"\n  Кадр {frame_idx}: выходит за пределы данных (ball_positions: {len(ball_positions)}, player_positions: {len(player_positions)})")
                i += 1
                continue
            
            best_idx, best_player_id, best_distance = self.find_best_consecutive_candidate(
                i, candidate_frames, ball_positions, player_positions
            )
            
            if best_idx == -1 or best_player_id == -1:
                if debug:
                    print(f"\n  Кадр {frame_idx}: не удалось найти подходящего кандидата")
                i += 1
                continue
            
            # Получаем позиции для лучшего кадра
            best_frame = candidate_frames[best_idx]
            ball_pos = self.get_ball_position(ball_positions, best_frame)
            player_pos = self.get_player_position(player_positions, best_frame, best_player_id)
            
            if ball_pos is None or player_pos is None:
                if debug:
                    print(f"\n  Кадр {best_frame}: нет данных о позициях")
                i = best_idx + 1
                continue
            
            # Считаем количество последовательных кадров с тем же игроком
            consecutive_count = best_idx - i + 1
            if consecutive_count > 1:
                if debug:
                    print(f"\n Найдено {consecutive_count} последовательных кадров с ближайшим игроком {best_player_id}")
                    print(f"   Лучший кадр: {best_frame} (расстояние: {best_distance:.1f}px)")
                for j in range(i, best_idx + 1):
                    f = candidate_frames[j]
                    p, d, _ = self.find_closest_player_in_frame(f, ball_positions, player_positions)
                    if debug:
                        status = "✓ ЛУЧШИЙ" if f == best_frame else ""
                        print(f"   Кадр {f}: игрок {p}, расстояние: {d:.1f}px {status}")
            
            # Определяем тип события для лучшего кадра
            event_type, event_reason = self.detect_event_type(
                best_frame, ball_pos, player_pos, best_player_id,
                candidate_frames, ball_positions, best_distance, player_positions
            )

            if debug:
                print(f"\n Кадр {best_frame} (выбран из {consecutive_count} последовательных):")
                print(f"   Позиция мяча: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
                print(f"   Тип события: {event_type.upper()} - {event_reason}")
            
            if event_type == 'unknown':
                i = best_idx + 1
                continue
            
            if self.game_state['last_shot_frame'] > 0:
                frame_diff = best_frame - self.game_state['last_shot_frame']
                if frame_diff < self.MIN_SHOT_INTERVAL:
                    if debug:
                        print(f"    Пропускаем: слишком близко к предыдущему событию ({frame_diff} кадров < {self.MIN_SHOT_INTERVAL})")
                    i = best_idx + 1
                    continue
            
            confidence = min(1.0, 1.0 - best_distance / self.PROXIMITY_THRESHOLD)
            
            event = ShotEvent(
                frame=best_frame,
                player_id=best_player_id,
                event_type=event_type,
                ball_position=ball_pos,
                confidence=confidence,
                timestamp=best_frame / self.fps
            )
            
            if event_type == 'serve':
                self.handle_serve_event(event)
            elif event_type == 'shot':
                self.handle_shot_event(event)
            elif event_type == 'bounce':
                self.stats['bounces'] += 1
                if debug:
                    print(f"    Отскок зарегистрирован на кадре {best_frame}")
            
            if event_type in ['serve', 'shot']:
                detected_events.append(event)
                self.shots_by_frame[best_frame] = event
                
                self.update_statistics(event)
                
                self.game_state['last_shot_frame'] = best_frame
                self.game_state['last_shot_player'] = best_player_id
                self.game_state['last_event_type'] = event_type
                
                if debug:
                    print(f"    ДОБАВЛЕНО: {event_type.upper()} - Игрок {best_player_id} (расстояние: {best_distance:.1f}px, уверенность: {confidence:.2f})")
            elif event_type == 'bounce':
                self.game_state['last_event_type'] = event_type
            
            self.debug_info.append({
                'frame': best_frame,
                'event_type': event_type,
                'player_id': best_player_id,
                'distance': best_distance,
                'reason': event_reason,
                'ball_position': ball_pos,
                'consecutive_count': consecutive_count,
                'was_best_of_consecutive': consecutive_count > 1
            })
            
            i = best_idx + 1
        
        if self.current_rally and not self.current_rally.is_completed:
            last_frame = len(ball_positions) - 1
            self.end_current_rally(last_frame)
        
        for rally in self.rallies:
            if rally.is_completed:
                self.update_serve_statistics(rally)
        
        if debug:
            self.print_debug_summary()
        
        return detected_events
    
    def handle_serve_event(self, event: ShotEvent, debug=False):
        """Обрабатывает событие подачи"""
        self.stats['serves'] += 1
        self.stats['player_stats'][event.player_id]['serves'] += 1
        
        self.game_state['serve_count_in_rally'] += 1
        
        if self.current_rally and self.game_state['serve_count_in_rally'] > 1:
            if debug:
                print(f"  ВТОРАЯ подача в розыгрыше #{self.current_rally.rally_id}")
            self.current_rally.is_double_fault = True
        
        if self.current_rally is None:
            self.start_new_rally(event.frame, event)
            self.game_state['serve_detected'] = True
            self.game_state['current_server'] = event.player_id
        else:
            if debug:
                print(f" Новая подача в середине розыгрыша, завершаем текущий")
            self.update_serve_statistics(self.current_rally)
            self.end_current_rally(event.frame - 1)
            self.start_new_rally(event.frame, event)
    
    def handle_shot_event(self, event: ShotEvent, debug=False):
        """Обрабатывает событие удара"""
        self.stats['shots'] += 1
        self.stats['player_stats'][event.player_id]['shots'] += 1
        
        if self.current_rally is None:
            if debug:
                print(f"  Удар без активного розыгрыша, начинаем новый")
            self.start_new_rally(event.frame)
        
        event.rally_id = self.current_rally.rally_id
        self.current_rally.shots.append(event)
    
    def update_statistics(self, event: ShotEvent):
        """Обновляет статистику"""
        self.stats['total_events'] += 1
    
    def get_statistics(self) -> Dict:
        """Возвращает полную статистику"""
        rally_stats = {
            'total_rallies': len([r for r in self.rallies if r.is_completed]),
            'ongoing_rallies': len([r for r in self.rallies if not r.is_completed]),
            'total_shots': sum(len(r.shots) for r in self.rallies),
            'average_shots_per_rally': 0,
            'longest_rally': 0,
        }
        
        completed_rallies = [r for r in self.rallies if r.is_completed]
        if completed_rallies:
            rally_lengths = [r.shot_count for r in completed_rallies]
            rally_stats['average_shots_per_rally'] = np.mean(rally_lengths)
            rally_stats['longest_rally'] = max(rally_lengths)
        
        return {
            'events': self.stats,
            'rallies': rally_stats,
            'players': self.stats['player_stats'],
            'serve_stats': self.serve_stats,
            'shots_by_frame': self.shots_by_frame,
            'all_rallies': self.rallies,
        }
    
    def save_detailed_statistics(self, filename="output_videos/game_statistics.txt"):
        stats = self.get_statistics()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(" СТАТИСТИКА ИГРЫ\n")
            
            f.write(f"\n Всего событий: {stats['events']['total_events']}\n")
            f.write(f"   Подач: {stats['events']['serves']}\n")
            f.write(f"   Ударов: {stats['events']['shots']}\n")
            
            f.write(f"\n Статистика по игрокам:\n")
            for player_id in [1, 2]:
                p_stats = stats['players'][player_id]
                f.write(f"   Игрок {player_id}: {p_stats['shots']} ударов, "
                        f"{p_stats['serves']} подач\n")
            
            f.write(f"\n СТАТИСТИКА ПОДАЧ:")
            for player_id in [1, 2]:
                serve_stats = stats['serve_stats'][player_id]
                f.write(f"\n   Игрок {player_id}:\n")
                f.write(f"      Всего подач: {serve_stats.total_serves}\n")
                f.write(f"      Первая подача: {serve_stats.first_serves_in}/{serve_stats.first_serves} "
                        f"({serve_stats.first_serve_percentage:.1f}%)\n")
                f.write(f"      Вторая подача: {serve_stats.second_serves}\n")
                f.write(f"      Успешных подач: {serve_stats.successful_serves} "
                        f"({serve_stats.serve_success_percentage:.1f}%)\n")
                f.write(f"      Двойных ошибок: {serve_stats.double_faults} "
                        f"({serve_stats.double_fault_percentage:.1f}%)\n")
            
            f.write(f"\n Статистика розыгрышей:\n")
            f.write(f"   Завершено розыгрышей: {stats['rallies']['total_rallies']}\n")
            f.write(f"   Всего ударов в розыгрышах: {stats['rallies']['total_shots']}\n")
            if stats['rallies']['total_rallies'] > 0:
                f.write(f"   Среднее ударов на розыгрыш: "
                        f"{stats['rallies']['average_shots_per_rally']:.1f}\n")
                f.write(f"   Самый длинный розыгрыш: {stats['rallies']['longest_rally']} ударов\n")
    
    def print_debug_summary(self):
        """Выводит итоговую отладочную информацию"""
        print(" ИТОГОВАЯ ОТЛАДОЧНАЯ ИНФОРМАЦИЯ")
        
        if not self.debug_info:
            print("Нет отладочной информации")
            return
        
        events_by_type = defaultdict(list)
        for info in self.debug_info:
            events_by_type[info['event_type']].append(info)
        
        print(f"\nВсего проанализировано кандидатов: {len(self.debug_info)}")
        
        for event_type, events in events_by_type.items():
            print(f"\n{event_type.upper()} ({len(events)}):")
            for i, event in enumerate(events[:10]):  # Показываем первые 10
                cons_info = f" [из {event['consecutive_count']} подряд]" if event.get('was_best_of_consecutive', False) else ""
                print(f"  {i+1}. Кадр {event['frame']}: Игрок {event['player_id']}, "
                      f"расстояние: {event['distance']:.1f}px{cons_info}")
                print(f"     Причина: {event['reason']}")
        
        if len(self.debug_info) > 10:
            print(f"\n... и еще {len(self.debug_info) - 10} событий")