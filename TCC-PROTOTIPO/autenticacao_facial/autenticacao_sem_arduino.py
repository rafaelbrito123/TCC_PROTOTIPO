import sys
import os
from deepface import DeepFace
import numpy as np
import cv2
from scipy.spatial.distance import cosine
import threading
import tkinter as tk
import time
import pygame
from PIL import ImageFont, ImageDraw, Image

# Diretórios do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import EMBEDDINGS_DIR

SOM_LIGANDO = "TCC_PROTOTIPO\\sons\\carro_ligando.wav"
SOM_LIGADO = "TCC_PROTOTIPO\\sons\\carro_ligado.wav"
SOM_DESLIGANDO = "TCC_PROTOTIPO\\sons\\carro_desligando.wav"
ALARME_SOM = "TCC_PROTOTIPO\\sons\\alarm.wav"

# Inicializa som
pygame.mixer.init()

def tocar_alarme(loop=False):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(ALARME_SOM)
    pygame.mixer.music.play(-1 if loop else 0)
    print("[ALARME] Som de alarme tocando...")

def parar_alarme():
    pygame.mixer.music.stop()
    print("[ALARME] Som de alarme parado.")

def tocar_som(caminho, loop=False):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play(-1 if loop else 0)
    print(f"[AUDIO] Tocando: {caminho}")

def parar_som():
    pygame.mixer.music.stop()
    print("[AUDIO] Som parado")

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
        representacoes = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)
        if not representacoes:
            return "nenhum_rosto"
        resultado = representacoes[0]["embedding"]
    except Exception as e:
        if "Face could not be detected" in str(e):
            return "nenhum_rosto"
        return None  # rosto detectado, mas erro ao gerar embedding

    menor_distancia = float("inf")
    usuario_mais_proximo = None

    for usuario, embeddings in usuarios_embeddings.items():
        for emb_salvo in embeddings:
            distancia = cosine(resultado, emb_salvo)
            if distancia < menor_distancia:
                menor_distancia = distancia
                usuario_mais_proximo = usuario

    if menor_distancia < limiar:
        return usuario_mais_proximo
    else:
        return None

# Função simulada para ligar/desligar o carro
def iniciar_simulador(nome_usuario):
    nome_usuario = nome_usuario.replace("_", " ").title()

    # Criar interface sem depender de Arduino
    root = tk.Tk()
    root.title(f"Simulador Automotivo - Bem-vindo, {nome_usuario}")
    root.geometry("600x480")
    root.configure(bg="black")

    status_label = tk.Label(root, text=f"Bem-vindo, {nome_usuario}! Pressione o botão de partida.", fg="cyan", bg="black", font=("Helvetica", 16))
    status_label.pack(pady=30)

    def atualizar_interface(texto, cor="white"):
        status_label.config(text=texto, fg=cor)
        root.update_idletasks()

    def fechar():
        parar_som()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", fechar)

    def escutar_simulacao():
        while True:
            try:
                # Simulando o botão de ligar e desligar
                entrada = input("Digite 'LIGAR' para ligar o carro ou 'desligado' para desligá-lo: ").strip()
                if entrada == "LIGAR":
                    threading.Thread(target=sequencia_ligacao).start()
                elif entrada == "desligado":
                    parar_som()
                    tocar_som(SOM_DESLIGANDO)
                    atualizar_interface(f"{nome_usuario}, o carro foi desligado.", "red")
                    time.sleep(2.0)
                    fechar()
            except Exception as e:
                print("[ERRO SIMULACAO]", e)
                break

    def sequencia_ligacao():
        atualizar_interface("Carro ligando...", "yellow")
        tocar_som(SOM_LIGANDO)
        time.sleep(2.8)
        tocar_som(SOM_LIGADO, loop=True)
        atualizar_interface("Carro ligado", "green")

    threading.Thread(target=escutar_simulacao, daemon=True).start()
    root.mainloop()

# Autenticação Facial
usuarios_embeddings = carregar_todos_embeddings()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

falhas_consecutivas = 0
limite_falhas = 5
alarme_soou = False
print("📷 Autenticando... Pressione 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    usuario_autenticado = autenticar_usuario(frame, usuarios_embeddings)

    if usuario_autenticado == "nenhum_rosto":
        texto = "Aguardando rosto..."
        cor = (200, 200, 200)
        falhas_consecutivas = 0

    elif usuario_autenticado:
        texto = f"Autenticado: {usuario_autenticado}"
        frame = draw_text_with_pil(frame, texto, (15, 10), font_size=30, color=(0, 255, 0))
        cv2.imshow("Autenticação Facial", frame)
        print(f"✅ Usuário {usuario_autenticado} autenticado!")
        time.sleep(1.5)
        break

    else:
        falhas_consecutivas += 1
        texto = "Rosto não reconhecido"
        cor = (255, 0, 0)
        print(f"[FALHA] Tentativa {falhas_consecutivas}/{limite_falhas}")

        if falhas_consecutivas >= limite_falhas and not alarme_soou:
            print("🚨 Tentativas excessivas! Alarme ativado.")
            tocar_alarme(loop=True)
            texto = "ALERTA! Tentativas inválidas"
            cor = (0, 0, 255)
            alarme_soou = True

    frame = draw_text_with_pil(frame, texto, (15, 10), font_size=30, color=cor)
    cv2.imshow("Autenticação Facial (Teste)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

parar_alarme()
cap.release()
cv2.destroyAllWindows()

if usuario_autenticado and usuario_autenticado != "nenhum_rosto":
    iniciar_simulador(usuario_autenticado)
