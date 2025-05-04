import cv2
import numpy as np
import os
from deepface import DeepFace

# Pasta onde estão os embeddings
EMBEDDINGS_PATH = r'D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\embeddings'

# Carregar os embeddings
def carregar_usuarios_embeddings():
    usuarios_embeddings = {}
    for usuario_folder in os.listdir(EMBEDDINGS_PATH):
        usuario_path = os.path.join(EMBEDDINGS_PATH, usuario_folder)
        if os.path.isdir(usuario_path):
            embeddings_list = []
            for file in os.listdir(usuario_path):
                if file.endswith('.npy'):
                    embedding = np.load(os.path.join(usuario_path, file))
                    embeddings_list.append(embedding)
            if embeddings_list:
                usuarios_embeddings[usuario_folder] = embeddings_list
    return usuarios_embeddings

usuarios_embeddings = carregar_usuarios_embeddings()

# Iniciar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Gerar embedding diretamente do frame inteiro (sem recortar rosto)
        embedding_frame = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )[0]["embedding"]

        usuario_autenticado = None

        for usuario, embeddings in usuarios_embeddings.items():
            for embedding_salvo in embeddings:
                distance = np.linalg.norm(np.array(embedding_frame) - np.array(embedding_salvo))
                if distance < 0.4:
                    usuario_autenticado = usuario
                    break
            if usuario_autenticado:
                break

        if usuario_autenticado:
            cv2.putText(frame, f"Autenticado: {usuario_autenticado}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Nao autenticado", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        cv2.putText(frame, "Sem rosto detectado", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Autenticacao Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
