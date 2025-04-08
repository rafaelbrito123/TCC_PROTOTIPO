import cv2
import os

# Caminho para os vídeos
videos_dir = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\datasets\drozy\videos_i8"
output_dir = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\datasets\drozy\frames"

# Cria pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Lista os vídeos
for video_file in os.listdir(videos_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        subject_name = os.path.splitext(video_file)[0]
        subject_dir = os.path.join(output_dir, subject_name)
        os.makedirs(subject_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{subject_name}_frame{frame_count:04}.jpg"
            frame_path = os.path.join(subject_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()
        print(f"Extraído {frame_count} frames de {video_file}")