import cv2
import numpy as np
from deepface import DeepFace
import os
import time

# Caminho da pasta de embeddings
EMBEDDINGS_DIR = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\embeddings"
THRESHOLD = 0.42  # Limite ajustado para permitir maior margem de erro
MAX_TRIES = 3  # Número máximo de tentativas de reconhecimento antes de considerar como "Desconhecido"

# Função para calcular distância do cosseno
def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Carrega todos os embeddings salvos
embeddings = {}
print(">> Carregando embeddings salvos...")
for arquivo in os.listdir(EMBEDDINGS_DIR):
    if arquivo.endswith(".npy"):
        nome = os.path.splitext(arquivo)[0]
        caminho = os.path.join(EMBEDDINGS_DIR, arquivo)
        embeddings[nome] = np.load(caminho)
print(f">> {len(embeddings)} usuário(s) carregado(s): {list(embeddings.keys())}")

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(">> Pressione 'q' para sair.")

frame_count = 0
check_interval = 30  # A cada 30 frames
usuario_identificado = "Verificando..."
tries = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    frame_count += 1
    embeddings_frame = []

    if frame_count % check_interval == 0:
        try:
            # Captura um único frame para gerar o embedding
            cv2.imwrite("frame_temp.jpg", frame)
            embedding_frame = DeepFace.represent("frame_temp.jpg", enforce_detection=True)[0]["embedding"]

            menor_distancia = float("inf")
            usuario_identificado = "Desconhecido"

            # Calcula a distância entre o embedding da imagem atual e os embeddings salvos
            for nome, embedding_salvo in embeddings.items():
                dist = cosine_distance(embedding_salvo, embedding_frame)
                if dist < menor_distancia:
                    menor_distancia = dist
                    if dist < THRESHOLD:
                        usuario_identificado = nome
                        tries = 0  # Resetando as tentativas de falha

            if usuario_identificado == "Desconhecido":
                tries += 1
                if tries >= MAX_TRIES:
                    usuario_identificado = "Tentativas esgotadas, desconhecido"

            print(f"[{time.strftime('%H:%M:%S')}] Distância: {menor_distancia:.4f} | Resultado: {usuario_identificado}")
        except Exception as e:
            usuario_identificado = "Erro na verificacao"
            print("Erro:", e)

    cor = (0, 255, 0) if usuario_identificado != "Desconhecido" else (0, 0, 255)
    cv2.putText(frame, usuario_identificado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.imshow("Autenticacao Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
