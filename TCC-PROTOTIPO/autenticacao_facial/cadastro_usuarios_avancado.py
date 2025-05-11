import os
import sys
import cv2
import numpy as np
import time
import winsound
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import EMBEDDINGS_DIR


def draw_text_with_pil(frame, text, position=(20, 50), font_size=24, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def desenhar_barra_progresso(frame, progresso, total=5):
    largura_barra = 200
    altura_barra = 20
    x, y = 20, frame.shape[0] - 40
    progresso_largura = int((progresso / total) * largura_barra)
    cv2.rectangle(frame, (x, y), (x + largura_barra, y + altura_barra), (180, 180, 180), 2)
    cv2.rectangle(frame, (x, y), (x + progresso_largura, y + altura_barra), (0, 255, 0), -1)
    return frame

# Entrada do nome
if len(sys.argv) > 1:
    nome_usuario = sys.argv[1].strip().lower().replace(" ", "_")
else:
    nome_usuario = input("Digite o nome do usuário: ").strip().lower().replace(" ", "_")
caminho_embedding = os.path.join(EMBEDDINGS_DIR, f"{nome_usuario}.npy")

instrucoes = [
    "Enquadre o rosto na moldura oval",
    "Aproxime um pouco o rosto",
    "Afaste um pouco o rosto"
]

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Largura
cap.set(4, 480)  # Altura

for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("❌ Webcam não pôde ser iniciada.")
    sys.exit(1)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

moldura_path = os.path.join(os.path.dirname(__file__), "imagens", "guia_oval.png")
if not os.path.exists(moldura_path):
    print(f"❌ Moldura não encontrada em {moldura_path}.")
    sys.exit(1)
moldura = cv2.imread(moldura_path, cv2.IMREAD_UNCHANGED)
moldura_resized = None

embeddings = []

print("📸 Iniciando cadastro... Pressione 'q' para sair.")

tolerancia = 15

for instrucao in instrucoes:
    print(f">> {instrucao}")
    capturado = False
    tempo_inicio = time.time()
    largura_anterior = None
    texto_dinamico = "Posicione o rosto..."

    progresso = 0
    capturas_por_instrucao = 5

    while progresso < capturas_por_instrucao:
        ret, frame = cap.read()
        if not ret:
            continue

        if moldura_resized is None:
            moldura_resized = cv2.resize(moldura, (frame.shape[1], frame.shape[0]))

        frame_overlay = frame.copy()
        if moldura_resized.shape[2] == 4:
            alpha_s = moldura_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame_overlay[:, :, c] = (alpha_s * moldura_resized[:, :, c] +
                                          alpha_l * frame_overlay[:, :, c])

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, 1.1, 5)

        if time.time() - tempo_inicio > 2:
            for (x, y, w, h) in rostos:
                cx, cy = x + w // 2, y + h // 2
                frame_h, frame_w = frame.shape[:2]
                margin_w, margin_h = int(frame_w * 0.3), int(frame_h * 0.3)
                center_x1 = (frame_w - margin_w) // 2
                center_y1 = (frame_h - margin_h) // 2
                center_x2 = center_x1 + margin_w
                center_y2 = center_y1 + margin_h

                if center_x1 <= cx <= center_x2 and center_y1 <= cy <= center_y2:
                    if largura_anterior is not None:
                        diferenca = w - largura_anterior
                        if diferenca > tolerancia:
                            texto_dinamico = "Você se aproximou"
                        elif diferenca < -tolerancia:
                            texto_dinamico = "Você se afastou"
                        else:
                            texto_dinamico = f"Capturando ({progresso+1}/{capturas_por_instrucao})..."
                            rosto_crop = frame[y:y+h, x:x+w]
                            try:
                                embedding = DeepFace.represent(rosto_crop, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                                embeddings.append(embedding)
                                progresso += 1
                                winsound.Beep(1000, 200)  # Beep de 200ms
                                print(f"✅ Captura {progresso}/{capturas_por_instrucao}")
                                time.sleep(0.2)
                            except Exception as e:
                                print(f"❌ Erro ao gerar embedding: {e}")
                                continue
                    largura_anterior = w

        frame_com_texto = draw_text_with_pil(frame_overlay.copy(), f"{instrucao} | {texto_dinamico}", (10, 10), 18, (255, 255, 0))
        frame_com_barra = desenhar_barra_progresso(frame_com_texto, progresso, capturas_por_instrucao)
        cv2.imshow("Cadastro Facial", frame_com_barra)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

cap.release()
cv2.destroyAllWindows()

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