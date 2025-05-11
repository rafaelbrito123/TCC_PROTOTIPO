import sys
import os
from deepface import DeepFace
import numpy as np
import cv2
from scipy.spatial.distance import cosine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import EMBEDDINGS_DIR
import subprocess
import serial
import time
from PIL import ImageFont, ImageDraw, Image

# Ajuste aqui sua porta COM
PORTA_ARDUINO = 'COM3'  # Substitua pela porta correta
arduino = serial.Serial(PORTA_ARDUINO, 9600, timeout=1)
time.sleep(2)  # Aguarda o Arduino reiniciar

def draw_text_with_pil(frame, text, position=(20, 50), font_size=32, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

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

def autenticar_usuario(frame, usuarios_embeddings, limiar=0.45):
    try:
        resultado = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
    except Exception:
        return None

    menor_distancia = float("inf")
    usuario_mais_proximo = None

    for usuario, embeddings in usuarios_embeddings.items():
        for emb_salvo in embeddings:
            distancia = cosine(resultado, emb_salvo)
            if distancia < menor_distancia:
                menor_distancia = distancia
                usuario_mais_proximo = usuario

    if menor_distancia < limiar:
        print(f"[DEBUG] Menor distância: {menor_distancia:.4f} (usuário: {usuario_mais_proximo})")
        return usuario_mais_proximo
    else:
        print(f"[DEBUG] Nenhum usuário abaixo do limiar. Menor distância: {menor_distancia:.4f}")
        return None


# Carrega os embeddings
usuarios_embeddings = carregar_todos_embeddings()

# Inicia webcam
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
        print(f"✅ Usuário {usuario_autenticado} autenticado!")



        cap.release()
        cv2.destroyAllWindows()
        arduino.close()

        # Inicia a simulação do carro
        subprocess.run([sys.executable, "TCC-PROTOTIPO\simulador\simulador_carro_ui.py", usuario_autenticado])
        break

    else:
        texto = "Rosto não reconhecido"
        cor = (255, 0, 0)

    frame = draw_text_with_pil(frame, texto, (15, 10), font_size=30, color=cor)
    cv2.imshow("Autenticação Facial (Teste)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()