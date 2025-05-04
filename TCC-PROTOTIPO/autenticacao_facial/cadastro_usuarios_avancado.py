import cv2
import numpy as np
import os
import sys
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
from config import EMBEDDINGS_DIR
import time

def draw_text_with_pil(frame, text, position=(20, 50), font_size=24, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

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
cap.set(3, 320)
cap.set(4, 240)

for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("❌ Webcam não pôde ser iniciada.")
    sys.exit(1)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

moldura_path = os.path.join(r'D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\imagens\guia_oval.png')
if not os.path.exists(moldura_path):
    print(f"❌ Moldura não encontrada em {moldura_path}.")
    sys.exit(1)
moldura = cv2.imread(moldura_path, cv2.IMREAD_UNCHANGED)
moldura_resized = None

embeddings = []

print("📸 Iniciando cadastro... Pressione 'q' para sair.")

for instrucao in instrucoes:
    largura_anterior = None
tolerancia = 15  # Tolerância em pixels para considerar como "estável"

for instrucao in instrucoes:
    print(f">> {instrucao}")
    capturado = False
    tempo_inicio = time.time()
    largura_anterior = None
    texto_dinamico = "Posicione o rosto..."

    while not capturado:
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
                frame_overlay[:, :, c] = (alpha_s * moldura_resized[:, :, c] + alpha_l * frame_overlay[:, :, c])

        frame_com_texto = draw_text_with_pil(frame_overlay.copy(), f"{instrucao} | {texto_dinamico}", (10, 10), 18, (255, 255, 0))

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
                            texto_dinamico = "Ok! Capturando..."
                            cv2.imshow("Cadastro Facial", draw_text_with_pil(frame_overlay.copy(), texto_dinamico, (10, 10)))
                            cv2.waitKey(1000)
                            try:
                                rosto_crop = frame[y:y+h, x:x+w]
                                embedding = DeepFace.represent(rosto_crop, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                                embeddings.append(embedding)
                                print("✅ Captura registrada.")
                                capturado = True
                                break
                            except Exception as e:
                                print(f"❌ Erro ao gerar embedding: {e}")
                                continue
                    largura_anterior = w

        cv2.imshow("Cadastro Facial", frame_com_texto)
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
