from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from trackers import BallTrackerNew, PlayerTrackerNew, CourtDetectorNew, NetDetector
from utils import measure_distance, get_center_of_bbox, get_foot_position, draw_player_stats, convert_pixel_distance_to_meters
from copy import deepcopy
import sys
import time
import math
import constants
from mini_court import MiniCourtWithNetDivision, CourtVisualizerWithNet
from statistics import create_fullscreen_mini_court_video, save_fullscreen_summary_image, SpeedCalculator

from shot_detector import SimpleShotDetector, ShotEvent, Rally

import matplotlib.pyplot as plt

def plot_extremums_analysis(ball_positions, extremums, output_path="extremums_analysis.png"):
    """График с отмеченными экстремумами"""
    # Получаем Y координаты
    y_values = []
    for pos in ball_positions:
        if pos and len(pos) == 4:
            y1, y2 = pos[1], pos[3]
            y_values.append((y1 + y2) / 2)
        else:
            y_values.append(np.nan)
    
    # Интерполируем
    y_values = pd.Series(y_values).interpolate().tolist()
    
    plt.figure(figsize=(14, 8))
    
    # 1. График Y координаты
    plt.subplot(2, 1, 1)
    plt.plot(range(len(y_values)), y_values, 'b-', linewidth=1.5, alpha=0.7, label='Y координата')
    
    # Отмечаем экстремумы
    maxima_frames = [e['frame'] for e in extremums if e['type'] == 'maximum']
    maxima_y = [y_values[f] for f in maxima_frames if f < len(y_values)]
    
    minima_frames = [e['frame'] for e in extremums if e['type'] == 'minimum']
    minima_y = [y_values[f] for f in minima_frames if f < len(y_values)]
    
    plt.scatter(maxima_frames, maxima_y, color='red', s=80, 
                marker='v', label=f'Максимумы ({len(maxima_frames)})', zorder=5)
    plt.scatter(minima_frames, minima_y, color='green', s=80,
                marker='^', label=f'Минимумы ({len(minima_frames)})', zorder=5)
    
    plt.xlabel('Кадр')
    plt.ylabel('Y координата')
    plt.title('Экстремумы траектории мяча')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. График производной
    plt.subplot(2, 1, 2)
    
    if len(y_values) > 1:
        delta_y = np.diff(y_values)
        delta_y = np.concatenate([[0], delta_y])
        
        plt.plot(range(len(delta_y)), delta_y, 'g-', linewidth=1.5, alpha=0.7, label='Производная ΔY')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Отмечаем точки смены знака
        sign_changes = []
        for i in range(1, len(delta_y)):
            if delta_y[i-1] * delta_y[i] < 0:  # Смена знака
                sign_changes.append(i)
        
        if sign_changes:
            change_y = [delta_y[i] for i in sign_changes]
            plt.scatter(sign_changes, change_y, color='orange', s=50,
                       label=f'Смены знака ({len(sign_changes)})', zorder=5)
        
        plt.xlabel('Кадр')
        plt.ylabel('ΔY (производная)')
        plt.title('Производная с точками смены знака')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"График экстремумов сохранен: {output_path}")

def plot_ball_movement(ball_positions, output_path="ball_movement_analysis.png", 
                       title="Анализ движения мяча на мини-корте"):
    """
    Создает комплексный график анализа движения мяча по осям X и Y
    """
    if not ball_positions:
        print("Нет данных о позициях мяча для построения графика")
        return
    
    # Собираем данные
    frames = []
    x_values = []
    y_values = []
    
    for i, ball_dict in enumerate(ball_positions):
        if ball_dict and 1 in ball_dict and ball_dict[1] is not None:
            pos = ball_dict[1]
            if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                x, y = float(pos[0]), float(pos[1])
                frames.append(i)
                x_values.append(x)
                y_values.append(y)
    
    if not frames:
        print("Не удалось собрать данные для графика")
        return
    
    print(f"  Анализ {len(frames)} кадров с мячом")
    
    # Создаем большой график
    plt.figure(figsize=(18, 12))
    
    # 1. График координат X и Y
    plt.subplot(3, 3, 1)
    plt.plot(frames, x_values, 'b-', linewidth=2, label='X координата', alpha=0.7)
    plt.plot(frames, y_values, 'r-', linewidth=2, label='Y координата', alpha=0.7)
    plt.xlabel('Номер кадра')
    plt.ylabel('Координаты (пиксели)')
    plt.title('Координаты мяча по осям X и Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Статистика координат
    x_min, x_max = min(x_values), max(x_values)
    x_range = x_max - x_min
    x_mean, x_std = np.mean(x_values), np.std(x_values)
    
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    y_mean, y_std = np.mean(y_values), np.std(y_values)
    
    # 2. График только X координаты
    plt.subplot(3, 3, 2)
    plt.plot(frames, x_values, 'b-', linewidth=2, label='X координата')
    plt.xlabel('Номер кадра')
    plt.ylabel('X координата (пиксели)')
    plt.title(f'X координата мяча\nДиапазон: {x_min:.1f} - {x_max:.1f}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=x_mean, color='r', linestyle='--', alpha=0.7, label=f'Среднее: {x_mean:.1f}')
    plt.fill_between(frames, x_mean - x_std, x_mean + x_std, alpha=0.2, color='b', label='±1 std')
    plt.legend()
    
    # 3. График только Y координаты
    plt.subplot(3, 3, 3)
    plt.plot(frames, y_values, 'r-', linewidth=2, label='Y координата')
    plt.xlabel('Номер кадра')
    plt.ylabel('Y координата (пиксели)')
    plt.title(f'Y координата мяча\nДиапазон: {y_min:.1f} - {y_max:.1f}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=y_mean, color='r', linestyle='--', alpha=0.7, label=f'Среднее: {y_mean:.1f}')
    plt.fill_between(frames, y_mean - y_std, y_mean + y_std, alpha=0.2, color='r', label='±1 std')
    plt.legend()
    
    # 4. График изменения X между кадрами
    plt.subplot(3, 3, 4)
    if len(x_values) > 1:
        x_changes = [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))]
        plt.plot(frames[1:], x_changes, 'b-', linewidth=1.5, label='|ΔX|', alpha=0.7)
        plt.xlabel('Номер кадра')
        plt.ylabel('|ΔX| (пиксели/кадр)')
        plt.title('Абсолютное изменение X координаты')
        plt.grid(True, alpha=0.3)
        
        if x_changes:
            x_mean_change = np.mean(x_changes)
            x_max_change = max(x_changes)
            x_threshold = x_mean_change * 2
            
            plt.axhline(y=x_mean_change, color='g', linestyle='--', alpha=0.7,
                       label=f'Среднее: {x_mean_change:.1f}')
            plt.axhline(y=x_threshold, color='orange', linestyle='--', alpha=0.7,
                       label=f'Порог: {x_threshold:.1f}')
            
            # Аномалии по X
            x_anomaly_frames = []
            x_anomaly_changes = []
            for i, change in enumerate(x_changes):
                if change > x_threshold:
                    x_anomaly_frames.append(frames[i+1])
                    x_anomaly_changes.append(change)
            
            if x_anomaly_frames:
                plt.scatter(x_anomaly_frames, x_anomaly_changes, color='blue', s=50,
                           zorder=5, label=f'Аномалии ({len(x_anomaly_frames)} шт)', alpha=0.6)
            
            plt.legend()
    
    # 5. График изменения Y между кадрами
    plt.subplot(3, 3, 5)
    if len(y_values) > 1:
        y_changes = [abs(y_values[i] - y_values[i-1]) for i in range(1, len(y_values))]
        plt.plot(frames[1:], y_changes, 'r-', linewidth=1.5, label='|ΔY|', alpha=0.7)
        plt.xlabel('Номер кадра')
        plt.ylabel('|ΔY| (пиксели/кадр)')
        plt.title('Абсолютное изменение Y координаты')
        plt.grid(True, alpha=0.3)
        
        if y_changes:
            y_mean_change = np.mean(y_changes)
            y_max_change = max(y_changes)
            y_threshold = y_mean_change * 2
            
            plt.axhline(y=y_mean_change, color='g', linestyle='--', alpha=0.7,
                       label=f'Среднее: {y_mean_change:.1f}')
            plt.axhline(y=y_threshold, color='orange', linestyle='--', alpha=0.7,
                       label=f'Порог: {y_threshold:.1f}')
            
            # Аномалии по Y
            y_anomaly_frames = []
            y_anomaly_changes = []
            for i, change in enumerate(y_changes):
                if change > y_threshold:
                    y_anomaly_frames.append(frames[i+1])
                    y_anomaly_changes.append(change)
            
            if y_anomaly_frames:
                plt.scatter(y_anomaly_frames, y_anomaly_changes, color='red', s=50,
                           zorder=5, label=f'Аномалии ({len(y_anomaly_frames)} шт)', alpha=0.6)
            
            plt.legend()
    
    # 6. График отношения изменений Y/X (для обнаружения отскоков)
    plt.subplot(3, 3, 6)
    if len(x_changes) > 0 and len(y_changes) > 0:
        # Вычисляем отношение Y/X для каждого кадра
        ratios = []
        valid_frames = []
        
        for i in range(min(len(x_changes), len(y_changes))):
            if x_changes[i] > 0:  # Избегаем деления на 0
                ratio = y_changes[i] / x_changes[i]
                ratios.append(ratio)
                valid_frames.append(frames[i+1])
        
        if ratios:
            plt.plot(valid_frames, ratios, 'g-', linewidth=1.5, label='Y/X отношение', alpha=0.7)
            plt.xlabel('Номер кадра')
            plt.ylabel('Отношение ΔY/ΔX')
            plt.title('Отношение изменений Y к X\n(>1 = преобладание вертикального движения)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Порог Y/X=1')
            plt.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Порог Y/X=2 (отскок?)')
            
            # Подсвечиваем возможные отскоки (Y/X > 2)
            bounce_frames = []
            bounce_ratios = []
            for i, ratio in enumerate(ratios):
                if ratio > 2:
                    bounce_frames.append(valid_frames[i])
                    bounce_ratios.append(ratio)
            
            if bounce_frames:
                plt.scatter(bounce_frames, bounce_ratios, color='red', s=60,
                           zorder=5, label=f'Возможные отскоки ({len(bounce_frames)} шт)')
            
            plt.legend()
    
    # 7. Траектория мяча (X-Y scatter plot)
    plt.subplot(3, 3, 7)
    scatter = plt.scatter(x_values, y_values, c=frames, cmap='viridis', 
                         s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
    plt.xlabel('X координата (пиксели)')
    plt.ylabel('Y координата (пиксели)')
    plt.title('Траектория мяча (цвет = номер кадра)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Номер кадра')
    
    # Добавляем стрелки для направления движения
    if len(x_values) > 5:
        step = len(x_values) // 5
        for i in range(0, len(x_values)-1, step):
            if i+1 < len(x_values):
                dx = x_values[i+1] - x_values[i]
                dy = y_values[i+1] - y_values[i]
                plt.arrow(x_values[i], y_values[i], dx*0.8, dy*0.8,
                         head_width=3, head_length=5, fc='red', ec='red', alpha=0.5)
    
    # 8. Гистограмма распределения координат
    plt.subplot(3, 3, 8)
    plt.hist(x_values, bins=30, alpha=0.5, color='blue', label='Распределение X')
    plt.hist(y_values, bins=30, alpha=0.5, color='red', label='Распределение Y')
    plt.xlabel('Координаты (пиксели)')
    plt.ylabel('Частота')
    plt.title('Распределение координат')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 9. Сводная статистика
    plt.subplot(3, 3, 9)
    plt.axis('off')  # Отключаем оси
    
    # Формируем текст статистики
    stats_text = f"=== СТАТИСТИКА ДВИЖЕНИЯ МЯЧА ===\n\n"
    stats_text += f"Всего кадров: {len(frames)}\n"
    stats_text += f"Кадры с мячом: {len(x_values)}\n\n"
    
    stats_text += f"=== КООРДИНАТЫ X ===\n"
    stats_text += f"Диапазон: {x_min:.1f} - {x_max:.1f} (Δ={x_range:.1f})\n"
    stats_text += f"Среднее: {x_mean:.1f} ± {x_std:.1f}\n"
    if 'x_mean_change' in locals():
        stats_text += f"Ср. изменение: {x_mean_change:.1f}/кадр\n"
        stats_text += f"Макс. изменение: {x_max_change:.1f}/кадр\n"
        stats_text += f"Аномалий X: {len(x_anomaly_frames) if 'x_anomaly_frames' in locals() else 0}\n\n"
    
    stats_text += f"=== КООРДИНАТЫ Y ===\n"
    stats_text += f"Диапазон: {y_min:.1f} - {y_max:.1f} (Δ={y_range:.1f})\n"
    stats_text += f"Среднее: {y_mean:.1f} ± {y_std:.1f}\n"
    if 'y_mean_change' in locals():
        stats_text += f"Ср. изменение: {y_mean_change:.1f}/кадр\n"
        stats_text += f"Макс. изменение: {y_max_change:.1f}/кадр\n"
        stats_text += f"Аномалий Y: {len(y_anomaly_frames) if 'y_anomaly_frames' in locals() else 0}\n\n"
    
    if 'ratios' in locals() and ratios:
        avg_ratio = np.mean(ratios)
        max_ratio = max(ratios)
        stats_text += f"=== АНАЛИЗ ОТНОШЕНИЙ ===\n"
        stats_text += f"Среднее Y/X: {avg_ratio:.2f}\n"
        stats_text += f"Макс. Y/X: {max_ratio:.2f}\n"
        if 'bounce_frames' in locals():
            stats_text += f"Возможных отскоков: {len(bounce_frames)}\n"
    
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'{title}\n', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Оставляем место для заголовка
    
    # Сохраняем график
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  График сохранен: {output_path}")
    
    # Возвращаем статистику
    stats = {
        'total_frames': len(frames),
        'frames_with_ball': len(x_values),
        
        # X статистика
        'x_min': x_min,
        'x_max': x_max,
        'x_range': x_range,
        'x_mean': x_mean,
        'x_std': x_std,
        'x_mean_change': x_mean_change if 'x_mean_change' in locals() else None,
        'x_max_change': x_max_change if 'x_max_change' in locals() else None,
        'x_anomalies_count': len(x_anomaly_frames) if 'x_anomaly_frames' in locals() else 0,
        
        # Y статистика
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_range,
        'y_mean': y_mean,
        'y_std': y_std,
        'y_mean_change': y_mean_change if 'y_mean_change' in locals() else None,
        'y_max_change': y_max_change if 'y_max_change' in locals() else None,
        'y_anomalies_count': len(y_anomaly_frames) if 'y_anomaly_frames' in locals() else 0,
        
        # Анализ отношений
        'avg_yx_ratio': avg_ratio if 'avg_ratio' in locals() else None,
        'max_yx_ratio': max_ratio if 'max_ratio' in locals() else None,
        'possible_bounces': len(bounce_frames) if 'bounce_frames' in locals() else 0,
    }
    
    return stats

def plot_ball_shot_analysis(ball_positions, candidate_frames, detected_shots, 
                           output_path="ball_shot_analysis.png",
                           title="Анализ детекции ударов по координате Y"):
    """
    Создает график для анализа детекции ударов:
    - Координата Y мяча
    - Кандидаты на удары (изменение направления)
    - Обнаруженные удары (после фильтрации)
    """
    if not ball_positions:
        print("Нет данных о позициях мяча для построения графика")
        return
    
    # Собираем данные Y координат
    frames = []
    y_values = []
    x_values = []
    
    for i, ball_pos in enumerate(ball_positions):
        if ball_pos is not None and len(ball_pos) == 4:
            # Получаем центр bounding box
            x1, y1, x2, y2 = ball_pos
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            frames.append(i)
            y_values.append(center_y)
            x_values.append(center_x)
    
    if not frames:
        print("Не удалось собрать данные для графика")
        return
    
    print(f"  Анализ {len(frames)} кадров для графика ударов")
    
    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Координата Y мяча с кандидатами и ударами
    ax1 = axes[0, 0]
    ax1.plot(frames, y_values, 'b-', linewidth=1.5, label='Y координата', alpha=0.7)
    ax1.set_xlabel('Номер кадра')
    ax1.set_ylabel('Y координата (пиксели)')
    ax1.set_title('Координата Y мяча с детекцией ударов')
    ax1.grid(True, alpha=0.3)
    
    # Отмечаем кандидатов (изменения направления)
    if candidate_frames:
        cand_y = [y_values[frame] if frame < len(y_values) else 0 for frame in candidate_frames]
        ax1.scatter(candidate_frames, cand_y, color='orange', s=40, 
                   label=f'Кандидаты ({len(candidate_frames)})', 
                   zorder=5, alpha=0.6, edgecolors='black')
    
    # Отмечаем обнаруженные удары
    if detected_shots:
        shot_frames = [shot['frame'] for shot in detected_shots]
        shot_y = [y_values[frame] if frame < len(y_values) else 0 for frame in shot_frames]
        
        # Разделяем подачи и обычные удары
        serve_frames = [shot['frame'] for shot in detected_shots if shot.get('is_serve', False)]
        serve_y = [y_values[frame] if frame < len(y_values) else 0 for frame in serve_frames]
        
        normal_shot_frames = [shot['frame'] for shot in detected_shots if not shot.get('is_serve', False)]
        normal_shot_y = [y_values[frame] if frame < len(y_values) else 0 for frame in normal_shot_frames]
        
        if serve_frames:
            ax1.scatter(serve_frames, serve_y, color='gold', s=100, 
                       marker='*', label=f'Подачи ({len(serve_frames)})',
                       zorder=6, edgecolors='black', linewidth=1)
        
        if normal_shot_frames:
            ax1.scatter(normal_shot_frames, normal_shot_y, color='red', s=80,
                       marker='o', label=f'Удары ({len(normal_shot_frames)})',
                       zorder=6, edgecolors='black', linewidth=1)
    
    ax1.legend(loc='upper right')
    
    # 2. Производная (изменение Y) с порогом
    ax2 = axes[0, 1]
    
    if len(y_values) > 1:
        # Вычисляем delta_y
        delta_y = np.diff(y_values)
        delta_y = np.concatenate([[0], delta_y])  # Добавляем 0 в начало
        
        # # Сглаживаем производную
        # window_size = 5
        # if len(delta_y) > window_size:
        #     delta_y_smooth = np.convolve(delta_y, np.ones(window_size)/window_size, mode='same')
        # else:
        #     delta_y_smooth = delta_y
        
        ax2.plot(frames, delta_y, 'g-', linewidth=1.5, label='ΔY (сглаженная)', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Пороги для обнаружения изменений
        threshold = np.std(delta_y) * 0.5
        ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Порог: {threshold:.1f}')
        ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        # Отмечаем моменты пересечения порога
        positive_crossings = []
        negative_crossings = []
        
        for i in range(1, len(delta_y)):
            if delta_y[i-1] > -threshold and delta_y[i] < -threshold:
                negative_crossings.append(i)
            elif delta_y[i-1] < threshold and delta_y[i] > threshold:
                positive_crossings.append(i)
        
        if negative_crossings:
            neg_y = [delta_y[frame] for frame in negative_crossings]
            ax2.scatter(negative_crossings, neg_y, color='red', s=30, 
                       alpha=0.6, label='Смена направления')
        
        ax2.set_xlabel('Номер кадра')
        ax2.set_ylabel('ΔY (изменение Y)')
        ax2.set_title('Производная координаты Y (изменение направления)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
    
    # 3. Траектория X-Y с ударами
    ax3 = axes[1, 0]
    scatter = ax3.scatter(x_values, y_values, c=frames, cmap='viridis', 
                         s=10, alpha=0.5, edgecolors='none')
    
    # Отмечаем удары на траектории
    if detected_shots:
        for shot in detected_shots:
            frame = shot['frame']
            if frame < len(x_values) and frame < len(y_values):
                color = 'gold' if shot.get('is_serve', False) else 'red'
                marker = '*' if shot.get('is_serve', False) else 'o'
                size = 120 if shot.get('is_serve', False) else 80
                
                ax3.scatter(x_values[frame], y_values[frame], 
                          color=color, s=size, marker=marker,
                          edgecolors='black', linewidth=1.5,
                          zorder=10, alpha=0.9)
                
                # Подписываем номер игрока
                player_id = shot.get('player_id', 0)
                ax3.annotate(f'P{player_id}', 
                           (x_values[frame], y_values[frame]),
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('X координата (пиксели)')
    ax3.set_ylabel('Y координата (пиксели)')
    ax3.set_title('Траектория мяча с ударами')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Номер кадра')
    
    # 4. Статистика обнаружения
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Формируем текст статистики
    stats_text = "=== СТАТИСТИКА ДЕТЕКЦИИ УДАРОВ ===\n\n"
    stats_text += f"Всего кадров: {len(frames)}\n"
    stats_text += f"Кандидатов на удары: {len(candidate_frames)}\n"
    stats_text += f"Обнаружено ударов: {len(detected_shots)}\n\n"
    
    if detected_shots:
        # Статистика по игрокам
        player_stats = {}
        for shot in detected_shots:
            player_id = shot.get('player_id', 0)
            player_stats[player_id] = player_stats.get(player_id, 0) + 1
        
        stats_text += "Удары по игрокам:\n"
        for player_id, count in player_stats.items():
            stats_text += f"  Игрок {player_id}: {count} ударов\n"
        
        # Разделение на подачи/удары
        serves = [s for s in detected_shots if s.get('is_serve', False)]
        shots = [s for s in detected_shots if not s.get('is_serve', False)]
        
        stats_text += f"\nПодачи: {len(serves)}\n"
        stats_text += f"Обычные удары: {len(shots)}\n"
        
        # Средняя скорость
        if detected_shots:
            speeds = [s.get('speed_after', 0) for s in detected_shots]
            avg_speed = np.mean(speeds) if speeds else 0
            stats_text += f"\nСредняя скорость удара: {avg_speed:.1f} px/s\n"
    
    # Координаты Y
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    y_mean, y_std = np.mean(y_values), np.std(y_values)
    
    stats_text += f"\n=== АНАЛИЗ КООРДИНАТЫ Y ===\n"
    stats_text += f"Диапазон Y: {y_min:.1f} - {y_max:.1f}\n"
    stats_text += f"Среднее Y: {y_mean:.1f} ± {y_std:.1f}\n"
    stats_text += f"Размах Y: {y_range:.1f}\n"
    
    # Распределение кандидатов
    if candidate_frames:
        cand_intervals = []
        for i in range(1, len(candidate_frames)):
            cand_intervals.append(candidate_frames[i] - candidate_frames[i-1])
        
        if cand_intervals:
            avg_interval = np.mean(cand_intervals)
            stats_text += f"\nСредний интервал между кандидатами: {avg_interval:.1f} кадров\n"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'{title}\n', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Сохраняем график
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  График анализа ударов сохранен: {output_path}")
    
    # Дополнительный детальный график для первых 100 кадров
    if len(frames) > 100:
        plot_detail_analysis(frames[:100], y_values[:100], 
                           candidate_frames, detected_shots,
                           output_path.replace('.png', '_detail.png'))
    
    return {
        'total_frames': len(frames),
        'candidates': len(candidate_frames),
        'detected_shots': len(detected_shots),
        'y_stats': {
            'min': y_min,
            'max': y_max,
            'mean': y_mean,
            'std': y_std,
            'range': y_range
        }
    }

def plot_detail_analysis(frames, y_values, candidate_frames, detected_shots, output_path):
    """Детальный график для первых 100 кадров"""
    plt.figure(figsize=(12, 6))
    
    # Отбираем кандидаты и удары в диапазоне
    detail_candidates = [f for f in candidate_frames if f < 100]
    detail_shots = [s for s in detected_shots if s['frame'] < 100]
    
    plt.plot(frames, y_values, 'b-', linewidth=2, label='Y координата', alpha=0.8)
    
    # Кандидаты
    if detail_candidates:
        cand_y = [y_values[f] for f in detail_candidates]
        plt.scatter(detail_candidates, cand_y, color='orange', s=80,
                   label=f'Кандидаты ({len(detail_candidates)})', 
                   zorder=5, alpha=0.8, edgecolors='black')
    
    # Обнаруженные удары
    if detail_shots:
        for shot in detail_shots:
            frame = shot['frame']
            if frame < len(y_values):
                color = 'gold' if shot.get('is_serve', False) else 'red'
                marker = '*' if shot.get('is_serve', False) else 'o'
                size = 150 if shot.get('is_serve', False) else 100
                
                plt.scatter(frame, y_values[frame], 
                          color=color, s=size, marker=marker,
                          edgecolors='black', linewidth=2,
                          zorder=10, alpha=0.9)
                
                # Подпись
                player_id = shot.get('player_id', 0)
                shot_type = "SERVE" if shot.get('is_serve', False) else "SHOT"
                plt.annotate(f'{shot_type} P{player_id}', 
                           (frame, y_values[frame]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold')
    
    plt.xlabel('Номер кадра (первые 100)')
    plt.ylabel('Y координата (пиксели)')
    plt.title('Детальный анализ детекции ударов (первые 100 кадров)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Детальный график сохранен: {output_path}")

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
    input_video_path = "input_videos/try.mp4"
    output_video_path = "output_videos/output_test.avi"
    
    print("\n Инициализация моделей...")
    ball_tracker = BallTrackerNew(model_path="models/ball.pt")
    player_tracker = PlayerTrackerNew(model_path="models/player.pt")
    court_detector = CourtDetectorNew(
        model_path="models/court", 
        use_refine_kps=False, 
        use_net_mask=True
    )
    net_model_path = "models/net"
    net_detector = NetDetector(model_path=net_model_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f" Ошибка открытия видео: {input_video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создание выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f" Ошибка создания видеофайла")
        return
    
    print(f" Выходной файл: {output_video_path}")
    
    # ========== ПЕРВЫЙ ПРОХОД: сбор детекций ==========
    print(f"\n Первый проход: сбор детекций...")
    
    frame_count = 0
    start_time = time.time()

    cap_first = cv2.VideoCapture(input_video_path)
    success, first_frame = cap_first.read()
    width_frame = first_frame.shape[1]
    cap_first.release()
    
    if not success:
        print(" Не удалось прочитать первый кадр!")
        return
    
    # Детекция корта на первом кадре
    first_frame_court_kps = court_detector.detect_keypoints(first_frame)
    # first_frame_court_kps = court_detector.refine_keypoints(first_frame, first_frame_court_kps)
    print(f" Корт найден: {first_frame_court_kps.shape}")
    
    # ДЕБАГ: сохраним точки для проверки
    debug_frame = first_frame.copy()
    for i, point in enumerate(first_frame_court_kps):
        if len(point) >= 2:
            x, y = float(point[0]), float(point[1])
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(debug_frame, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.putText(debug_frame, str(i), (int(x)+12, int(y)+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite("debug_court_points.jpg", debug_frame)
    
    # Создаем мини-корт
    print(f"\n Инициализация мини-корта с сеткой...")
    mini_court = MiniCourtWithNetDivision(first_frame, net_model_path)
    net_bbox = net_detector.detect_net_bbox(first_frame)
    
    print(f"   Сетка обнаружена: {'Да' if net_bbox is not None else 'Нет'}")
    if net_bbox is not None:
        print(f"   Координаты сетки: {net_bbox}")
    
    # Инициализируем мини-корт с сеткой
    net_line = mini_court.initialize_with_frame(first_frame, first_frame_court_kps, net_bbox)
    
    # ДОБАВЛЯЕМ НЕДОСТАЮЩИЙ МЕТОД split_areas_with_net В CourtAreaAnalyzerWithNet
    if not hasattr(mini_court.area_analyzer, 'split_areas_with_net'):
        def split_areas_with_net(court_keypoints):
            """Разделяет зоны на верхние и нижние части с помощью линии сетки"""
            if mini_court.area_analyzer.net_line is None:
                mini_court.area_analyzer.areas = mini_court.area_analyzer.base_areas.copy()
                print(" Сетка не определена, используем базовые зоны")
                return
            
            net_y = mini_court.area_analyzer.net_line[1]
            print(f" Разделение зон сеткой на Y={net_y}")
            
            # Получаем все полигоны с разделением
            polygon_list = mini_court.area_analyzer.get_polygon_coords(court_keypoints, net_y)
            
            # Обновляем self.areas для быстрого доступа
            mini_court.area_analyzer.areas = {}
            for poly_info in polygon_list:
                mini_court.area_analyzer.areas[poly_info['name']] = poly_info['indices']
            
            print(f" Создано {len(mini_court.area_analyzer.areas)} зон с разделение сеткой")
        
        mini_court.area_analyzer.split_areas_with_net = lambda court_kps: split_areas_with_net(court_kps, first_frame, width_frame)
    
    # Принудительно вызываем разделение зон
    if net_line is not None:
        print(f"\n Разделение зон сеткой...")
        if hasattr(mini_court.area_analyzer, 'split_areas_with_net'):
            mini_court.area_analyzer.split_areas_with_net(first_frame_court_kps, first_frame, width_frame)
        else:
            print(" Метод split_areas_with_net не найден, используем базовые зоны")
            mini_court.area_analyzer.areas = mini_court.area_analyzer.base_areas.copy()
        
        # Получаем полигоны с разделением для отладки
        if hasattr(mini_court.area_analyzer, 'get_polygon_coords'):
            polygon_list = mini_court.area_analyzer.get_polygon_coords(first_frame_court_kps, first_frame, width_frame, net_line[1])
            print(f"   Получено {len(polygon_list)} полигонов с разделение сеткой")
    else:
        print(" Сетка не обнаружена, используем базовые зоны")
        mini_court.area_analyzer.areas = mini_court.area_analyzer.base_areas.copy()
    
    # Создаем визуализатор
    visualizer = CourtVisualizerWithNet(mini_court.area_analyzer)
    
    # Сохраняем отладочный кадр с зонами
    try:
        debug_frame = visualizer.draw_court_with_net(
            first_frame.copy(),
            first_frame_court_kps,
            net_line
        )
        cv2.imwrite("debug_court_with_net_division.jpg", debug_frame)
        print(" Сохранен debug_court_with_net_division.jpg")
    except Exception as e:
        print(f" Ошибка при сохранении отладочного кадра: {e}")
    
    # ПЕРВЫЙ ПРОХОД: сбор данных
    print(f"\n Первый проход: сбор данных...")
    cap = cv2.VideoCapture(input_video_path)
    all_ball_detections = []
    all_player_detections = []
    all_foot_positions = []
    track_ball_history = defaultdict(list)
    player_tracks = defaultdict(list)
    all_court_keypoints = []
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Детекция мяча
        best_ball = ball_tracker.get_best_detection(frame)
        all_ball_detections.append(best_ball)
        
        # Детекция игроков
        player_dets = player_tracker.get_player_detections(frame)
        all_player_detections.append(player_dets)

        # Вычисляем foot_position
        foot_positions_frame = {}
        for tid, box in player_dets.items():
            foot_pos = get_foot_position(box)
            foot_positions_frame[tid] = foot_pos
            cx, cy = get_center_of_bbox(box)
            player_tracks[tid].append((frame_count, cx, cy))

        all_foot_positions.append(foot_positions_frame)
        
        # Сохраняем ключевые точки корта (используем те же, что и на первом кадре)
        all_court_keypoints.append(first_frame_court_kps.copy())
        
        frame_count += 1
        
        # Обновление прогресс-бара
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
    elapsed_time = time.time() - start_time
    print(f"\n Первый проход завершен за {elapsed_time:.1f} секунд")
    print(f"   Средняя скорость: {frame_count/elapsed_time:.1f} FPS")
    
    # ========== ОБРАБОТКА ДАННЫХ ==========
    print(f"\n Обработка данных...")
    
    # Интерполяция мяча
    print("   Интерполяция траектории мяча...")
    positions = ball_tracker.get_positions_for_interpolation(all_ball_detections)
    interpolated = ball_tracker.interpolate_positions(positions, max_gap=5)
    smoothed = ball_tracker.smooth_trajectory(interpolated, window_size=3)
    
    # Фильтрация игроков
    print("   Фильтрация игроков...")
    filtered_players = player_tracker.choose_and_filter_players_each_frame(all_court_keypoints, all_player_detections)

    print(f"Получили: ID игроков = {list(filtered_players[0].keys())}")

    # В основном коде, после создания mini_court:
    # Инициализация мини-корта с гомографией
    # mini_court = MiniCourtWithNetDivision(first_frame, net_model_path)

    # Инициализация гомографии
    if first_frame_court_kps is not None:
        # Преобразуем в нужный формат
        if isinstance(first_frame_court_kps, np.ndarray):
            court_kps_flat = first_frame_court_kps.flatten().tolist()
        else:
            court_kps_flat = first_frame_court_kps
        
        mini_court.initialize_homography_from_court_keypoints(court_kps_flat)

    # Конвертация через гомографию
    player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        filtered_players,
        interpolated,
        court_kps_flat
    )
    print(f"\n=== ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРА УДАРОВ ===")
    print(f"Используем исходные координаты для детекции ударов")
    
    # Конвертация координат для мини-корта
    print("   Конвертация координат для мини-корта...")
    
    # Нужно передать кадры для детекции сетки
    frames_for_net = []
    cap_temp = cv2.VideoCapture(input_video_path)
    for _ in range(min(10, total_frames)):  # Первые 10 кадров
        success, frame = cap_temp.read()
        if success:
            frames_for_net.append(frame)
    cap_temp.release()
    
    # try:
    #     print("   Используем конвертацию с разделением сеткой...")

    #     player_mini, ball_mini = mini_court.convert_with_net_division(
    #         filtered_players,
    #         interpolated,
    #         first_frame_court_kps,
    #         frames_for_net
    #     )

    #     print(f" Конвертация успешна: player_mini={len(player_mini)}, ball_mini={len(ball_mini)}")
    # except Exception as e:
    #     print(f" Ошибка конвертации с сеткой: {e}")
    #     print(f"Тип ошибки: {type(e).__name__}")
    #     print("   Используем упрощенную конвертацию...")
    #     court_kps_flat = first_frame_court_kps.flatten().tolist()
    #     player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
    #         filtered_players,
    #         interpolated,
    #         court_kps_flat
    #     )
    #     print(f" Упрощенная конвертация: player_mini={len(player_mini)}, ball_mini={len(ball_mini)}")
        
    print("   Создание графика сглаженного движения мяча...")
    smoothed_stats = plot_ball_movement(
        ball_mini,
        output_path="output_videos/ball_movement_mini.png",
        title="Изменение координаты мяча на мини корте"
    )

    # ========== ВТОРОЙ ПРОХОД: визуализация с расчетом скорости ==========
    print(f"\n Второй проход: визуализация с расчетом скорости...")
    
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0
    start_time = time.time()
    
    # Инициализация калькулятора скорости
    speed_calculator = SpeedCalculator(fps=fps, court_width_meters=constants.DOUBLE_LINE_WIDTH, court_length_meters=constants.COURT_HEIGHT)
    
    # Цвета для игроков
    player_colors = {
        1: (0, 255, 0),
        2: (255, 0, 0)
    }
    
    # История для сглаживания скоростей
    player_speed_histories = defaultdict(list)

    # player_stats_data = [{
    #     'frame_num': 0,
    #     'player_1_number_of_shots': 0,
    #     'player_1_total_shot_speed': 0,
    #     'player_1_last_shot_speed': 0,
    #     'player_2_number_of_shots': 0,
    #     'player_2_total_shot_speed': 0,
    #     'player_2_last_shot_speed': 0,
    # }]

    # # Получаем кадры с ударами мяча
    # ball_shot_frames = ball_tracker.get_ball_shot_frames(interpolated)

    # 1. Находим все экстремумы с помощью scipy
    print("Поиск экстремумов траектории с помощью scipy...")

    # Вариант A: Экстремумы Y координаты (рекомендуется)
    extremums_y = ball_tracker.get_all_extremums_scipy_simple(interpolated)
    candidate_frames_y = [e['frame'] for e in extremums_y]
    print(f"Вариант A (экстремумы Y): {len(candidate_frames_y)} кандидатов")

    # Выбираем какой вариант использовать (рекомендую вариант A)
    candidate_frames = candidate_frames_y  # или candidate_frames_deriv

    # 2. Создаем график анализа
    print("\nСоздание графиков анализа экстремумов...")

    # График для экстремумов Y
    if extremums_y:
        # Создаем свой график для scipy экстремумов
        y_values = []
        for ball_pos in interpolated:
            if ball_pos and len(ball_pos) == 4:
                y1, y2 = ball_pos[1], ball_pos[3]
                y_values.append((y1 + y2) / 2)
            else:
                y_values.append(np.nan)
        
        y_array = np.array(y_values, dtype=np.float64)
        mask = np.isnan(y_array)
        if np.any(mask):
            x = np.arange(len(y_array))
            y_array[mask] = np.interp(x[mask], x[~mask], y_array[~mask])
        
        plt.figure(figsize=(14, 8))
        
        # 1. График Y координаты
        plt.subplot(2, 1, 1)
        plt.plot(range(len(y_array)), y_array, 'b-', linewidth=1.5, alpha=0.7, label='Y координата')
        
        # Отмечаем экстремумы
        maxima_frames = [e['frame'] for e in extremums_y if e['type'] == 'maximum']
        maxima_y = [y_array[f] for f in maxima_frames if f < len(y_array)]
        
        minima_frames = [e['frame'] for e in extremums_y if e['type'] == 'minimum']
        minima_y = [y_array[f] for f in minima_frames if f < len(y_array)]
        
        plt.scatter(maxima_frames, maxima_y, color='red', s=80, 
                    marker='v', label=f'Максимумы ({len(maxima_frames)})', zorder=5)
        plt.scatter(minima_frames, minima_y, color='green', s=80,
                    marker='^', label=f'Минимумы ({len(minima_frames)})', zorder=5)
        
        plt.xlabel('Кадр')
        plt.ylabel('Y координата (пиксели)')
        plt.title('Экстремумы Y координаты мяча (scipy метод)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. График производной
        plt.subplot(2, 1, 2)
        derivation = np.diff(y_array)
        derivation = np.concatenate([[0], derivation])
        
        plt.plot(range(len(derivation)), derivation, 'g-', linewidth=1.5, alpha=0.7, label='Производная ΔY')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel('Кадр')
        plt.ylabel('ΔY (производная)')
        plt.title('Производная Y координаты')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("output_videos/ball_extremums_scipy_y.png", dpi=150)
        plt.close()
        
        print(f"График экстремумов scipy сохранен: output_videos/ball_extremums_scipy_y.png")

    # НОВЫЙ КОД (вставить вместо старого):
    print(f"\n{'='*60}")
    print("🎾 ИНИЦИАЛИЗАЦИЯ УЛУЧШЕННОГО ДЕТЕКТОРА УДАРОВ")
    print(f"{'='*60}")

    # 1. Инициализация нового детектора
    simple_shot_detector = SimpleShotDetector(fps=fps, drawing_rectangle_width=mini_court.court_drawing_width, drawing_rectangle_height=mini_court.court_drawing_height)

    # 2. Детекция событий (serve/shot/bounce)
    print(f"\n🔍 Детекция событий на мини-корте...")
    print(f"   Кандидатов для анализа: {len(candidate_frames)}")
    print(f"   Кадров с игроками: {len(player_mini)}")
    print(f"   Кадров с мячом: {sum(1 for b in ball_mini if b is not None)}")

    # 3. Запуск детекции
    detected_events = simple_shot_detector.detect_events(
        candidate_frames=candidate_frames,
        ball_positions=ball_mini,      # координаты мяча на мини-корте
        player_positions=player_mini   # координаты игроков на мини-корте
    )

    # 4. Получение результатов
    shots_by_frame = simple_shot_detector.shots_by_frame
    stats = simple_shot_detector.get_statistics()

    # 5. Вывод статистики
    simple_shot_detector.print_detailed_statistics()

    # 6. Сохранение статистики в файл
    try:
        with open('shot_detector_stats.txt', 'w', encoding='utf-8') as f:
            f.write("=== СТАТИСТИКА УДАРОВ И РОЗЫГРЫШЕЙ ===\n\n")
            f.write(f"Всего событий: {stats['events']['total_events']}\n")
            f.write(f"Подач: {stats['events']['serves']}\n")
            f.write(f"Ударов: {stats['events']['shots']}\n")
            f.write(f"Отскоков: {stats['events']['bounces']}\n\n")
            
            f.write("Статистика по игрокам:\n")
            for player_id in [1, 2]:
                p_stats = stats['players'][player_id]
                f.write(f"  Игрок {player_id}: {p_stats['shots']} ударов, {p_stats['serves']} подач\n")
            
            f.write(f"\nРозыгрыши:\n")
            f.write(f"  Завершено: {stats['rallies']['total_rallies']}\n")
            f.write(f"  Всего ударов в розыгрышах: {stats['rallies']['total_shots']}\n")
            if stats['rallies']['total_rallies'] > 0:
                f.write(f"  Среднее ударов на розыгрыш: {stats['rallies']['average_shots_per_rally']:.1f}\n")
                f.write(f"  Самый длинный розыгрыш: {stats['rallies']['longest_rally']} ударов\n")
            
            f.write(f"\nДетали по розыгрышам:\n")
            for i, rally in enumerate(simple_shot_detector.rallies[:10]):
                status = "завершен" if rally.is_completed else "в процессе"
                winner = f"Победитель: Игрок {rally.winner}" if rally.winner else "Без победителя"
                f.write(f"  #{i+1}: кадры {rally.start_frame}-{rally.end_frame or '?'}, "
                    f"{rally.shot_count} ударов ({status}) - {winner}\n")
        
        print(f"\n✅ Статистика сохранена в shot_detector_stats.txt")
    except Exception as e:
        print(f"\n⚠️ Ошибка при сохранении статистики: {e}")

    # 7. Подготовка данных для визуализации
    # Преобразуем события в старый формат для совместимости с графиками
    detected_shots_for_plots = []
    for event in detected_events:
        if event.event_type in ['serve', 'shot']:
            detected_shots_for_plots.append({
                'frame': event.frame,
                'player_id': event.player_id,
                'is_serve': event.event_type == 'serve',
                'hit_type': event.event_type,
                'confidence': event.confidence,
                'ball_position': event.ball_position,
                'speed_after': 0  # можно добавить если есть
            })

    print(f"\n📊 Подготовка к визуализации:")
    print(f"   Ударов для отображения: {len(shots_by_frame)}")
    print(f"   Всего событий: {len(detected_events)}")

    if detected_events:
        print(f"\nПримеры обнаруженных событий (первые 5):")
        for i, event in enumerate(detected_events[:5]):
            event_type_ru = "ПОДАЧА" if event.event_type == 'serve' else "УДАР" if event.event_type == 'shot' else "ОТСКОК"
            print(f"  {i+1}. {event_type_ru:8} - Игрок {event.player_id} на кадре {event.frame} "
                f"(уверенность: {event.confidence:.2f})")

    # 8. Создание графиков анализа
    print(f"\n{'='*60}")
    print("📈 СОЗДАНИЕ ГРАФИКОВ АНАЛИЗА")
    print(f"{'='*60}")

    # Создаем график с анализом
    shot_analysis_stats = plot_ball_shot_analysis(
        ball_positions=interpolated,
        candidate_frames=candidate_frames,
        detected_shots=detected_shots_for_plots,
        output_path="output_videos/ball_shot_analysis_new.png",
        title="Анализ детекции ударов по изменению координаты Y (новая логика)"
    )

    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*60}")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        annotated = frame.copy()

        try:
            # ===== ВИЗУАЛИЗАЦИЯ ЗОН КОРТА С СЕТКОЙ =====
            current_net_line = mini_court.net_line
            
            if current_net_line is not None:
                # Рисуем зоны корта с разделением сеткой
                annotated = visualizer.draw_court_with_net(
                    annotated,
                    first_frame_court_kps,
                    current_net_line
                )

            else:
                # Сетка не обнаружена, рисуем только точки
                for i, point in enumerate(first_frame_court_kps):
                    if len(point) >= 2:
                        x = float(point[0])
                        y = float(point[1])
                        if not np.isnan(x) and not np.isnan(y):
                            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
                            
        except Exception as e:
            if frame_count % 30 == 0:
                print(f" Ошибка отрисовки зон (кадр {frame_count}): {e}")
            for i, point in enumerate(first_frame_court_kps):
                if len(point) >= 2:
                    x = float(point[0])
                    y = float(point[1])
                    if not np.isnan(x) and not np.isnan(y):
                        cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # ===== ВИЗУАЛИЗАЦИЯ МЯЧА =====
        if frame_count < len(smoothed):
            pos = smoothed[frame_count]
            if pos is not None and not np.any(np.isnan(pos)):
                try:
                    x1, y1, x2, y2 = map(int, pos)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, "BALL", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    pass
        
        # ===== ВИЗУАЛИЗАЦИЯ ИГРОКОВ =====
        if frame_count < len(filtered_players):
            curr_players = filtered_players[frame_count]
            for permanent_id, box in curr_players.items():
                try:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Цвета в зависимости от постоянного ID
                    if permanent_id == 1:
                        color = (0, 255, 0)  # Зеленый для нижнего игрока
                    elif permanent_id == 2:
                        color = (255, 0, 0)  # Красный для верхнего игрока
                    else:
                        color = (255, 255, 0)  # Желтый для других
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"P{permanent_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except:
                    pass
        
        # ===== ВИЗУАЛИЗАЦИЯ МИНИ-КОРТА =====
        if mini_court:
                # Получаем позиции для текущего кадра
                player_pos = player_mini[frame_count] if frame_count < len(player_mini) else None
                ball_pos = ball_mini[frame_count] if frame_count < len(ball_mini) else None
                
                # ===== РАСЧЕТ СКОРОСТЕЙ =====
                player_speeds = {}
                
                if ball_pos is not None:
                    ball_pos_for_draw  = ball_pos        
                
                if player_pos is not None:
                    for player_id, position in player_pos.items():
                        if position is not None:
                            # Получаем bbox игрока из filtered_players
                            player_bbox = None
                            if frame_count < len(filtered_players):
                                curr_players = filtered_players[frame_count]
                                if player_id in curr_players:
                                    player_bbox = curr_players[player_id]
                            
                            player_speed_m_s, player_speed_kmh = speed_calculator.update_player_speed(
                                player_id, position, player_bbox=player_bbox
                            )
                            
                            # Сглаживаем скорость игрока
                            player_speed_histories[player_id].append(player_speed_kmh)
                            if len(player_speed_histories[player_id]) > 10:
                                player_speed_histories[player_id].pop(0)
                            
                            if player_speed_histories[player_id]:
                                player_speed_kmh = np.mean(player_speed_histories[player_id])
                            
                            player_speeds[player_id] = player_speed_kmh
                
                # Рисуем мини-корт
                annotated = mini_court.draw_mini_court_with_zones(
                    annotated,
                    player_positions=player_pos,
                    ball_positions=ball_pos_for_draw
                )

                # ===== ОТРИСОВКА УДАРОВ НА МИНИ-КОРТЕ =====
                if frame_count in shots_by_frame:
                    event = shots_by_frame[frame_count]
                    
                    # Визуализация в зависимости от типа события
                    if event.event_type == 'serve':
                        color = (0, 255, 255)  # желтый для подачи
                        label = f"SERVE P{event.player_id}"
                        marker_size = 15
                    elif event.event_type == 'shot':
                        color = (0, 255, 0)    # зеленый для удара
                        label = f"SHOT P{event.player_id}"
                        marker_size = 12
                    elif event.event_type == 'bounce':
                        color = (255, 0, 0)    # красный для отскока
                        label = "BOUNCE"
                        marker_size = 10
                    
                    # Рисуем маркер на мини-корте
                    if hasattr(mini_court, 'court_start_x'):
                        # Конвертируем координаты мини-корта в координаты основного кадра
                        court_x = mini_court.court_start_x
                        court_y = mini_court.court_start_y
                        
                        # Позиция на основном кадре
                        marker_x = int(court_x + event.ball_position[0])
                        marker_y = int(court_y + event.ball_position[1])
                        
                        # Рисуем круг
                        cv2.circle(annotated, (marker_x, marker_y), marker_size, color, 2)
                        
                        # Подпись
                        cv2.putText(annotated, label, (marker_x + 15, marker_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Для подачи добавляем дополнительный эффект
                        if event.event_type == 'serve':
                            cv2.circle(annotated, (marker_x, marker_y), marker_size + 5, color, 1)
                    
                    # Также рисуем маркер на основном кадре для лучшей видимости
                    if frame_count < len(smoothed):
                        ball_pos_main = smoothed[frame_count]
                        if ball_pos_main is not None and len(ball_pos_main) == 4:
                            x1, y1, x2, y2 = ball_pos_main
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            cv2.circle(annotated, (center_x, center_y), 10, color, 2)
                            
                            # Краткая подпись рядом с мячом
                            short_label = "S" if event.event_type == 'serve' else "H" if event.event_type == 'shot' else "B"
                            cv2.putText(annotated, f"{short_label}P{event.player_id}", 
                                    (center_x + 15, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Вывод информации в консоль (первые 10 событий)
                    if len(simple_shot_detector.shots) <= 10:
                        event_type_ru = "ПОДАЧА" if event.event_type == 'serve' else "УДАР" if event.event_type == 'shot' else "ОТСКОК"
                        print(f"Кадр {frame_count}: {event_type_ru} - Игрок {event.player_id}")
                
                # ===== ОТОБРАЖЕНИЕ СКОРОСТЕЙ НА ВИДЕО =====
                # 1. Панель скоростей в углу
                annotated = speed_calculator.draw_speed_info(
                    annotated, 
                    player_speeds,
                    position=(width - 350, height - 180)
                )

                if simple_shot_detector.current_rally:
                    rally = simple_shot_detector.current_rally
                    rally_text = f"Rally #{rally.rally_id}: {rally.shot_count} shots"
                    
                    # Позиция для отображения (верхний левый угол)
                    cv2.putText(annotated, rally_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated, f"Serving: P{rally.serving_player}" if rally.serving_player else "No serve",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        out.write(annotated)
        frame_count += 1
        
        # Обновление прогресс-бара
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

    print("СОЗДАНИЕ ВИДЕО С МИНИ-КОРТОМ И ТРАЕКТОРИЯМИ")
    
    fullscreen_video_path = "output_videos/fullscreen_mini_court_trajectories_try.avi"
    fullscreen_image_path = "output_videos/fullscreen_mini_court_summary_try.jpg"
    
    fullscreen_width = 1280
    fullscreen_height = 720
    
    # В основном коде, после расчета скоростей:
    create_fullscreen_mini_court_video(
        input_video_path=input_video_path,
        output_video_path=fullscreen_video_path,
        output_image_path=fullscreen_image_path,
        player_mini=player_mini,
        ball_mini=ball_mini,
        mini_court_data=mini_court,
        fps=fps,
        player_avg_heights=speed_calculator.player_avg_heights,  # Передаем средние высоты боксов!
        trajectory_length=30,
        width=fullscreen_width,
        height=fullscreen_height
    )
    
    print(f"\n Видео сохранено: {output_video_path}")
    
    print(f"\n=== СТАТИСТИКА СКОРОСТЕЙ ===")

    for player_id in [1, 2]:
        if player_speed_histories[player_id]:
            avg_speed = np.mean(player_speed_histories[player_id])
            max_speed = max(player_speed_histories[player_id])
            print(f"👤 Игрок {player_id}:")
            print(f"  Средняя скорость: {avg_speed:.1f} км/ч ({avg_speed/3.6:.1f} м/с)")
            print(f"  Максимальная скорость: {max_speed:.1f} км/ч ({max_speed/3.6:.1f} м/с)")

    # Добавьте статистику по розыгрышам
    print(f"\n🏓 Статистика розыгрышей:")
    for i, rally in enumerate(simple_shot_detector.rallies):
        status = "завершен" if rally.is_completed else "в процессе"
        winner_info = f"Победитель: P{rally.winner}" if rally.winner else "Без победителя"
        print(f"  #{i+1}: {rally.shot_count} ударов, {status}, {winner_info}")

if __name__ == "__main__":
    main()