import cv2
from deepface import DeepFace
import numpy as np
import os
import time
from config import EMBEDDINGS_DIR


os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Nome do usuário
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

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

embeddings = []
print(">> Iniciando captura automática...")

for instrucao in instrucoes:
    print(f">> {instrucao}")
    tempo_instrucao = 3
    inicio = time.time()

    while time.time() - inicio < tempo_instrucao:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            break

        frame_exibido = frame.copy()
        cv2.putText(frame_exibido, instrucao, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("Cadastro Facial Avançado", frame_exibido)

        cv2.waitKey(1)

    # Captura a imagem após o tempo da instrução
    try:
        embedding = DeepFace.represent(frame, enforce_detection=True)[0]["embedding"]
        embeddings.append(embedding)
        print("✅ Captura registrada.")
    except Exception as e:
        print("❌ Erro ao gerar embedding:", e)

# Encerra webcam
cap.release()
cv2.destroyAllWindows()

if embeddings:
    embedding_medio = np.mean(embeddings, axis=0)
    np.save(caminho_embedding, embedding_medio)
    print(f">> Cadastro finalizado com sucesso: {caminho_embedding}")
else:
    print("⚠ Nenhum embedding foi salvo.")
