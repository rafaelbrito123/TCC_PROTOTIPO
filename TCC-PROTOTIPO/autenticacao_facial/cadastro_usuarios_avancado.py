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

def desenhar_escurecimento_com_elipse(frame, centro, eixos):
    mask = np.zeros_like(frame, dtype=np.uint8)
    mask[:] = (0, 0, 0)
    mask = cv2.ellipse(mask, centro, eixos, 0, 0, 360, (255, 255, 255), -1)
    frame_masked = cv2.addWeighted(frame, 0.3, mask, 0.7, 0)
    frame_resultado = np.where(mask == 255, frame, frame_masked)
    return frame_resultado

def desenhar_barra_em_volta_da_elipse(frame, centro, eixos, progresso, total):
    angulo = int(360 * (progresso / total))
    overlay = frame.copy()
    cv2.ellipse(overlay, centro, eixos, 0, 0, angulo, (0, 255, 0), 6)
    return overlay

if len(sys.argv) > 1:
    nome_usuario = sys.argv[1].strip().lower().replace(" ", "_")
else:
    nome_usuario = input("Digite o nome do usuário: ").strip().lower().replace(" ", "_")

caminho_embedding = os.path.join(EMBEDDINGS_DIR, f"{nome_usuario}.npy")

instrucoes = [
    "Enquadre o rosto na moldura oval (distância média)",
    "Aproxime mais o rosto",
    "Afaste um pouco o rosto"
]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Aquece a câmera
for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("❌ Webcam não pôde ser iniciada.")
    sys.exit(1)

# Carrega modelo antecipadamente
print("⏳ Carregando modelo DeepFace...")
dummy = np.zeros((160, 160, 3), dtype=np.uint8)
_ = DeepFace.represent(dummy, model_name="Facenet", enforce_detection=False)
print("✅ Modelo carregado.")

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

embeddings = []

print("📸 Iniciando cadastro... Pressione 'q' para sair.")

for idx, etapa in enumerate(instrucoes):
    print(f">> {etapa}")
    progresso = 0
    capturas_por_instrucao = 5

    tempo_inicio = time.time()

    while progresso < capturas_por_instrucao:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        altura, largura = frame.shape[:2]
        centro = (largura // 2, altura // 2)

        if etapa == instrucoes[0]:  # distância média
            eixo_maior, eixo_menor = 120, 160
            min_face_width = eixo_menor * 1.4
            max_face_width = eixo_menor * 1.55
        elif etapa == instrucoes[1]:  # aproximar
            eixo_maior, eixo_menor = 140, 190
            min_face_width = eixo_menor * 1.5
            max_face_width = eixo_menor * 1.65
        else:  # afastar
            eixo_maior, eixo_menor = 90, 150
            min_face_width = eixo_menor * 1.3
            max_face_width = eixo_menor * 1.45

        frame = desenhar_escurecimento_com_elipse(frame, centro, (eixo_maior, eixo_menor))

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, 1.1, 5)

        texto_dinamico = "Centralize seu rosto na moldura."

        for (x, y, w, h) in rostos:
            cx, cy = x + w // 2, y + h // 2
            if abs(cx - centro[0]) < 40 and abs(cy - centro[1]) < 50:
                if w < min_face_width:
                    texto_dinamico = "Aproxime-se mais!"
                elif w > max_face_width:
                    texto_dinamico = "Afaste-se um pouco!"
                else:
                    texto_dinamico = f"Capturando ({progresso + 1}/{capturas_por_instrucao})..."
                    rosto_crop = frame[y:y + h, x:x + w]
                    try:
                        embedding = DeepFace.represent(rosto_crop, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                        embeddings.append(embedding)
                        progresso += 1
                        winsound.Beep(1000, 200)
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"❌ Erro ao gerar embedding: {e}")
                        continue

        frame = draw_text_with_pil(frame, etapa + " | " + texto_dinamico, (10, 10), 18, (255, 255, 0))
        frame = desenhar_barra_em_volta_da_elipse(frame, centro, (eixo_maior, eixo_menor), progresso, capturas_por_instrucao)
        cv2.imshow("Cadastro Facial", frame)

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
