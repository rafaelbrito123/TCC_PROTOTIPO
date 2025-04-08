import cv2
from deepface import DeepFace
import numpy as np
import time

# Caminho da imagem de referência
IMG_REFERENCIA = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\usuario1\referencia.jpeg"

# Gera embedding da imagem de referência
try:
    embedding_ref = DeepFace.represent(IMG_REFERENCIA, enforce_detection=True)[0]["embedding"]
except Exception as e:
    print("Erro ao gerar embedding da referência:", e)
    exit()

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Pressione 'q' para sair")

frame_count = 0
check_interval = 30
autenticado = False
ultimo_resultado = "Verificando..."

def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam")
        break

    frame_count += 1

    if frame_count % check_interval == 0:
        try:
            cv2.imwrite("frame_temp.jpg", frame)
            embedding_frame = DeepFace.represent("frame_temp.jpg", enforce_detection=True)[0]["embedding"]
            dist = cosine_distance(embedding_ref, embedding_frame)

            threshold = 0.35  # Ajustável. Quanto menor, mais rigoroso.
            autenticado = dist < threshold
            ultimo_resultado = "Autenticado" if autenticado else "Nao reconhecido"
            print(f"[{time.strftime('%H:%M:%S')}] Distância: {dist:.4f} | {ultimo_resultado}")
        except Exception as e:
            ultimo_resultado = "Erro na verificacao"
            print("Erro:", e)

    # Mostra resultado na tela
    cor = (0, 255, 0) if autenticado else (0, 0, 255)
    cv2.putText(frame, ultimo_resultado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.imshow("Autenticacao Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
