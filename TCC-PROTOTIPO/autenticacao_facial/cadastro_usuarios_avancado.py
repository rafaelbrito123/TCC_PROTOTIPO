from deepface import DeepFace
import numpy as np
import os
import time
import cv2
import sys
from config import EMBEDDINGS_DIR
from PIL import ImageFont, ImageDraw, Image

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Função para desenhar texto com acento usando PIL
def draw_text_with_pil(frame, text, position=(20, 50), font_size=24, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Nome do usuário
if len(sys.argv) > 1:
    nome_usuario = sys.argv[1].strip().lower().replace(" ", "_")
else:
    nome_usuario = input("Digite o nome do usuário: ").strip().lower().replace(" ", "_")

caminho_embedding = os.path.join(EMBEDDINGS_DIR, f"{nome_usuario}.npy")

# Instruções com acentos
instrucoes = [
    "Olhe para frente", "Sorria", "Vire o rosto levemente para a esquerda",
    "Vire o rosto levemente para a direita", "Olhe um pouco para cima",
    "Olhe um pouco para baixo", "Feche os olhos e abra",
    "Faça uma expressão séria", "Aproxime o rosto da câmera", "Afaste um pouco o rosto"
]

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# Pré-aquecimento da câmera (descarta os primeiros frames)
for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("❌ Webcam não pôde ser iniciada.")
    sys.exit(1)

embeddings = []
print("📸 Iniciando captura automática...")

for instrucao in instrucoes:
    print(f">> {instrucao}")
    inicio = time.time()
    tempo_instrucao = 2.5

    while time.time() - inicio < tempo_instrucao:
        ret, frame = cap.read()
        if not ret:
            print("❌ Não foi possível ler frame da webcam.")
            break

        exibido = draw_text_with_pil(frame.copy(), instrucao, position=(10, 10), font_size=22, color=(255, 255, 0))
        cv2.imshow("Cadastro Facial Avançado", exibido)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    # Captura frame final para gerar embedding
    ret, frame = cap.read()
    if ret:
        try:
            embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
            embeddings.append(embedding)
            print("✅ Captura registrada.")
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
    else:
        print("⚠ Frame final inválido.")

# Encerra webcam
cap.release()
cv2.destroyAllWindows()

# Salva embeddings
if embeddings:
    novo_embedding = np.array(embeddings)
    if os.path.exists(caminho_embedding):
        existente = np.load(caminho_embedding)
        if existente.ndim == 1:
            existente = np.expand_dims(existente, axis=0)
        novo_embedding = np.concatenate([existente, novo_embedding])
    np.save(caminho_embedding, novo_embedding)
    print(f"✅ Cadastro finalizado: {len(novo_embedding)} embeddings salvos para {nome_usuario}.")
else:
    print("⚠ Nenhum embedding foi salvo.")
