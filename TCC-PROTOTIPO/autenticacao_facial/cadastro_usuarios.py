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

# Instruções para capturas variadas
instrucoes = [
    "Posicione o rosto no centro com expressão neutra",
    "Aproxime o rosto da câmera",
    "Afaste um pouco o rosto",
    "Vire levemente o rosto para a esquerda",
    "Vire levemente o rosto para a direita",
    "Sorria levemente"
]

capturas = []

# Inicia webcam
cap = cv2.VideoCapture(0)
print(">> Aguarde... preparando captura automática de rostos")
time.sleep(2)

for instrucao in instrucoes:
    print(f"\n>> {instrucao}")
    tempo_instrucao = 4  # segundos para cada instrução
    inicio = time.time()

    while time.time() - inicio < tempo_instrucao:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Mostra instrução na tela
        frame_display = frame.copy()
        cv2.putText(frame_display, instrucao, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        segundos_restantes = tempo_instrucao - int(time.time() - inicio)
        cv2.putText(frame_display, f"Capturando em {segundos_restantes}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Cadastro Facial Automático", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cadastro cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Captura automática do frame
    try:
        print(">> Capturando e gerando embedding...")
        embedding = DeepFace.represent(frame, enforce_detection=True)[0]["embedding"]
        capturas.append(embedding)
    except Exception as e:
        print("Erro ao capturar embedding:", e)

cap.release()
cv2.destroyAllWindows()

if len(capturas) >= 3:
    media_embedding = np.mean(np.array(capturas), axis=0)
    np.save(caminho_embedding, media_embedding)
    print(f"\n✅ Cadastro concluído com sucesso! Embedding salvo em: {caminho_embedding}")
else:
    print("\n⚠ Cadastro incompleto. Foram capturadas menos de 3 expressões válidas.")