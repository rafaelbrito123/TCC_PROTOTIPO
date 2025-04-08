import cv2
import os
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from deepface import DeepFace
import logging

# Configurações para reduzir logs
os.environ['GLOG_minloglevel'] = '2'  # Silencia MediaPipe
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Silencia TensorFlow

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Landmarks dos olhos (índices do MediaPipe)
OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
OLHO_DIREITO = [33, 160, 158, 133, 153, 144]

def calcular_ear(pontos_olho, landmarks):
    """Calcula o Eye Aspect Ratio (EAR) para um olho."""
    coords = [(landmarks[i].x, landmarks[i].y) for i in pontos_olho]
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Limiares (ajuste conforme suas imagens!)
EAR_NORMAL = 0.23      # EAR >= 0.25 → olhos abertos
EAR_SEMIABERTO = 0.14  # 0.18 <= EAR < 0.25 → semiabertos
# EAR < 0.18 → olhos fechados

# Pasta com as imagens
pasta = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\teste_deepface\known_faces\rostos"

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(('.jpg', '.png')):
        try:
            # Carrega a imagem
            caminho_imagem = os.path.join(pasta, arquivo)
            img = cv2.imread(caminho_imagem)
            
            if img is None:
                print(f"Erro: Imagem não carregada - {arquivo}")
                continue

            print(f"\nAnalisando: {arquivo}")
            
            # Converte para RGB (MediaPipe requer RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks faciais
            resultados = face_mesh.process(img_rgb)
            
            if not resultados.multi_face_landmarks:
                print("  → Nenhum rosto detectado.")
                continue

            landmarks = resultados.multi_face_landmarks[0].landmark

            # Calcula EAR para ambos os olhos
            ear_esquerdo = calcular_ear(OLHO_ESQUERDO, landmarks)
            ear_direito = calcular_ear(OLHO_DIREITO, landmarks)
            ear_medio = (ear_esquerdo + ear_direito) / 2

            # Classificação baseada no EAR
            if ear_medio >= EAR_NORMAL:
                status = "NORMAL (olhos abertos)"
            elif ear_medio >= EAR_SEMIABERTO:
                status = "LEVE CANSACO (olhos semiabertos)"
            else:
                status = "SONOLÊNCIA (olhos fechados)"

            # Opcional: Adiciona análise de emoção (DeepFace)
            try:
                resultado = DeepFace.analyze(img, actions=("emotion"), silent=True, enforce_detection=False)
                emocao = resultado[0]["dominant_emotion"]
            except:
                emocao = "Não detectada"

            # Output no console
            print(f"  → EAR esquerdo: {ear_esquerdo:.3f} | direito: {ear_direito:.3f}")
            print(f"  → Status: {status}")
            print(f"  → Emoção predominante: {emocao}")

        except Exception as e:
            print(f"  → Erro ao processar {arquivo}: {str(e)}")