from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import os
import cv2
from config import EMBEDDINGS_DIR


def carregar_todos_embeddings():
    usuarios_embeddings = {}

    for arquivo in os.listdir(EMBEDDINGS_DIR):
        if arquivo.endswith(".npy"):
            nome = os.path.splitext(arquivo)[0]
            dados = np.load(os.path.join(EMBEDDINGS_DIR, arquivo))
            if dados.ndim == 1:
                dados = [dados]  # transforma em lista se for 1 embedding só
            usuarios_embeddings[nome] = dados

    return usuarios_embeddings


def avaliar_distancias(frame, usuarios_embeddings, modelo="Facenet", limiar=0.45):
    try:
        resultado = DeepFace.represent(frame, model_name=modelo, enforce_detection=True)[0]["embedding"]
    except Exception as e:
        print("❌ Erro ao gerar embedding da imagem:", e)
        return

    print("\n📊 Distâncias com os usuários salvos:")
    for usuario, embeddings in usuarios_embeddings.items():
        for i, emb_salvo in enumerate(embeddings):
            distancia = cosine(resultado, emb_salvo)
            status = "✅ abaixo do limiar" if distancia < limiar else "❌ acima do limiar"
            print(f"{usuario} [{i+1}] → Distância: {distancia:.4f} → {status}")


# Exemplo de uso direto no script:
if __name__ == "__main__":
    print(">> Iniciando avaliação de precisão facial...")

    embeddings_usuarios = carregar_todos_embeddings()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_exibido = frame.copy()
        cv2.putText(frame_exibido, "Aperte ESPACO p/ avaliar - ESC p/ sair", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Avaliador de Distancia", frame_exibido)

        tecla = cv2.waitKey(1)
        if tecla == 27:  # ESC
            break
        elif tecla == 32:  # Espaço
            avaliar_distancias(frame, embeddings_usuarios)

    cap.release()
    cv2.destroyAllWindows()
