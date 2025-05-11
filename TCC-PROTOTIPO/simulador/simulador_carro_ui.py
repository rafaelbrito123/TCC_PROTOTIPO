import tkinter as tk
import threading
import serial
import time
import pygame

# Caminhos dos áudios
som_ligando = "TCC-PROTOTIPO/sons/carro_ligando.wav"
som_ligado = "TCC-PROTOTIPO/sons/carro_ligado.wav"
som_desligando = "TCC-PROTOTIPO/sons/carro_desligando.wav"

# Inicia pygame mixer
pygame.mixer.init()

# Inicializa serial com o Arduino
arduino = serial.Serial('COM3', 9600, timeout=1)

# Funções de áudio
def tocar_som(caminho, loop=False):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play(-1 if loop else 0)
    print(f"[AUDIO] Tocando: {caminho}")

def parar_som():
    pygame.mixer.music.stop()
    print("[AUDIO] Som parado")

# Interface
root = tk.Tk()
root.title("Simulador Automotivo")
root.geometry("600x480")
root.configure(bg="black")

status_label = tk.Label(root, text="Aguardando autenticação...", fg="white", bg="black", font=("Helvetica", 16))
status_label.pack(pady=30)

def atualizar_interface(texto, cor="white"):
    status_label.config(text=texto, fg=cor)
    root.update_idletasks()
    print(f"[INTERFACE] {texto}")

# Fecha janela com segurança
def fechar():
    parar_som()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", fechar)

# Thread que escuta a porta serial
def escutar_serial():
    while True:
        try:
            if arduino.in_waiting > 0:
                linha = arduino.readline().decode().strip()
                if linha:
                    print(f"[SERIAL] {linha}")

                if linha == "LIGAR":
                    threading.Thread(target=sequencia_ligacao).start()

                elif linha == "desligado":
                    parar_som()
                    tocar_som(som_desligando)
                    atualizar_interface("Carro desligado", "red")
                    time.sleep(2.0)
                    root.destroy()
        except Exception as e:
            print("[ERRO SERIAL]", e)
            break

# Sequência de ligar carro
def sequencia_ligacao():
    atualizar_interface("Carro ligando...", "yellow")
    tocar_som(som_ligando)
    time.sleep(2.8)

    arduino.write(b"ATIVAR\n")  # Ativa o relé/LED
    tocar_som(som_ligado, loop=True)
    atualizar_interface("Carro ligado", "green")

# Simula reconhecimento facial
def simular_reconhecimento():
    time.sleep(2)
    atualizar_interface("Rosto reconhecido. Pressione o botão físico.", "cyan")
    arduino.write(b"PRONTO\n")

# Inicia threads
threading.Thread(target=escutar_serial, daemon=True).start()
threading.Thread(target=simular_reconhecimento, daemon=True).start()

# Inicia interface
root.mainloop()