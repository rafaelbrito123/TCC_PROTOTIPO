from deepface import DeepFace
import numpy as np
import os
import time
import cv2
import sys
from config import EMBEDDINGS_DIR
from PIL import ImageFont, ImageDraw, Image

# Carrega moldura oval
moldura = cv2.imread(r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\imagens\guia_oval.png", cv2.IMREAD_UNCHANGED)
moldura_resized = None  # Vamos redimensionar depois


# Cria pasta se não existir
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

# Instruções simplificadas
instrucoes = [
    "Enquadre o rosto na moldura oval",
    "Aproxime um pouco o rosto",
    "Afaste um pouco o rosto"
]

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# Pré-aquecimento da câmera
for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("❌ Webcam não pôde ser iniciada.")
    sys.exit(1)

# Inicializa detector de rosto
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

embeddings = []

print("📸 Iniciando cadastro... Pressione Q para sair.")

for instrucao in instrucoes:
    print(f">> {instrucao}")
    capturado = False

    while not capturado:
        ret, frame = cap.read()
        if not ret:
            print("❌ Não foi possível capturar da webcam.")
            continue

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5)
    

        # Sobrepõe moldura
        frame_overlay = frame.copy()
        if moldura_resized.shape[2] == 4:
            alpha_s = moldura_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame_overlay[:, :, c] = (alpha_s * moldura_resized[:, :, c] + alpha_l * frame_overlay[:, :, c])

        # Exibe instrução e moldura
        exibido = draw_text_with_pil(frame_overlay.copy(), instrucao, position=(10, 10), font_size=20, color=(255, 255, 0))
        cv2.imshow("Cadastro Facial", exibido)

        for (x, y, w, h) in rostos:
            # Verifica se o rosto está dentro da moldura central
            if 80 < x < 160 and 60 < y < 130:
                try:
                    embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                    embeddings.append(embedding)
                    print("✅ Captura registrada.")
                    capturado = True
                    break
                except Exception as e:
                    print(f"❌ Erro ao gerar embedding: {e}")
                    continue
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

# Encerra webcam
cap.release()
cv2.destroyAllWindows()

# Salva os embeddings
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
