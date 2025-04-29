import os

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, output_dir, frames_per_second=1):
    """
    Извлекает кадры из видео с указанной частотой

    Args:
        video_path: путь к видеофайлу
        output_dir: директория для сохранения кадров
        frames_per_second: количество кадров, извлекаемых за секунду видео
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Вычисляем шаг извлечения кадров
    step = int(fps / frames_per_second)

    current_frame = 0
    frame_number = 0

    # Получаем имя видеофайла без расширения
    video_name = os.path.basename(video_path).split(".")[0]

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Извлекаем каждый n-й кадр
        if current_frame % step == 0:
            frame_path = os.path.join(
                output_dir, f"{video_name}_frame_{frame_number:05d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frame_number += 1

        current_frame += 1

    video.release()
    return frame_number


def extract_all_frames(input_dir, output_dir, frames_per_second=10):
    """
    Извлекает кадры из всех видео в указанной директории
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".mkv") and f.startswith("Movie_")
    ]
    total_frames = 0

    for video_file in tqdm(video_files, desc="Извлечение кадров из видео"):
        video_path = os.path.join(input_dir, video_file)
        frames = extract_frames(video_path, output_dir, frames_per_second)
        total_frames += frames

    print(f"Всего извлечено {total_frames} кадров из {len(video_files)} видео")


extract_all_frames("pigs", "frames")
