from deepface import DeepFace
import numpy as np
import os
import cv2
from scipy.spatial.distance import cosine
from config import EMBEDDINGS_DIR

def carregar_todos_embeddings():
    usuarios_embeddings = {}
    for arquivo in os.listdir(EMBEDDINGS_DIR):
        if arquivo.endswith(".npy"):
            nome = arquivo.replace(".npy", "")
            caminho = os.path.join(EMBEDDINGS_DIR, arquivo)
            embeddings = np.load(caminho)
            if len(embeddings.shape) == 1:
                embeddings = np.expand_dims(embeddings, axis=0)
            usuarios_embeddings[nome] = embeddings
    return usuarios_embeddings

def autenticar_usuario(frame, usuarios_embeddings, limiar=0.54):
    try:
        resultado = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
    except Exception as e:
        return None

    for usuario, embeddings in usuarios_embeddings.items():
        for emb_salvo in embeddings:
            distancia = cosine(resultado, emb_salvo)
            if distancia < limiar:
                return usuario
    return None

# Carrega todos os embeddings salvos
usuarios_embeddings = carregar_todos_embeddings()

# Inicia a webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("📷 Modo de Teste de Autenticação Facial (pressione 'q' para sair)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    usuario_autenticado = autenticar_usuario(frame, usuarios_embeddings)

    if usuario_autenticado:
        texto = f"Autenticado: {usuario_autenticado}"
        cor = (0, 255, 0)
    else:
        texto = "Rosto não reconhecido"
        cor = (0, 0, 255)

    cv2.putText(frame, texto, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    cv2.imshow("Autenticação Facial (Teste)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
