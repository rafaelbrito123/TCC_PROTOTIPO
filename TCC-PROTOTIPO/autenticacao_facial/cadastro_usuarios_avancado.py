from deepface import DeepFace
import numpy as np
import os
import time
import cv2
from config import EMBEDDINGS_DIR

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

import sys

# Verifica se o nome do usuário foi passado como argumento
if len(sys.argv) > 1:
    nome_usuario = sys.argv[1].strip().lower().replace(" ", "_")
else:
    nome_usuario = input("Digite o nome do usuário: ").strip().lower().replace(" ", "_")

caminho_embedding = os.path.join(EMBEDDINGS_DIR, f"{nome_usuario}.npy")

# Instruções para o usuário
instrucoes = [
    "Olhe para frente",
    "Sorria",
    "Vire o rosto levemente para a esquerda",
    "Vire o rosto levemente para a direita",
    "Olhe um pouco para cima",
    "Olhe um pouco para baixo",
    "Feche os olhos e abra",
    "Faça uma expressão séria",
    "Aproxime o rosto da câmera",
    "Afaste um pouco o rosto"
]

# Inicia webcam com resolução menor (para suavizar o vídeo)
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # largura
cap.set(4, 240)  # altura

embeddings = []
print(">> Iniciando captura automática...")

for instrucao in instrucoes:
    print(f">> {instrucao}")
    tempo_instrucao = 3
    inicio = time.time()

    # Mostrar vídeo ao vivo com a instrução por alguns segundos
    while time.time() - inicio < tempo_instrucao:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            break

        frame_exibido = frame.copy()
        cv2.putText(frame_exibido, instrucao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.imshow("Cadastro Facial Avançado", frame_exibido)
        cv2.waitKey(10)

    # Captura frame final para o embedding
    ret, frame = cap.read()
    if ret:
        try:
            embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
            embeddings.append(embedding)
            print("✅ Captura registrada.")
        except Exception as e:
            print("❌ Erro ao gerar embedding:", e)

# Encerra webcam
cap.release()
cv2.destroyAllWindows()

# Salvar múltiplos embeddings do mesmo usuário (acumula no mesmo arquivo)
if embeddings:
    novo_embedding = np.array(embeddings)
    if os.path.exists(caminho_embedding):
        existente = np.load(caminho_embedding)
        if existente.ndim == 1:
            existente = np.expand_dims(existente, axis=0)
        novo_embedding = np.concatenate([existente, novo_embedding])
    np.save(caminho_embedding, novo_embedding)
    print(f">> Cadastro finalizado. {len(novo_embedding)} embeddings salvos para {nome_usuario}")
else:
    print("⚠ Nenhum embedding foi salvo.")
