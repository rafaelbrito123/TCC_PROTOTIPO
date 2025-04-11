import tkinter as tk
from tkinter import ttk
import sys
import time

usuario = sys.argv[1] if len(sys.argv) > 1 else "Usuário"

# Função para simular o "ligar o carro"
def ligar_carro():
    status_label.config(text="🔓 Rosto reconhecido com sucesso! Ligando o carro...", foreground="green")
    root.update()
    time.sleep(2)
    painel_label.config(text="🚗 Carro ligado! Bem-vindo, {}!".format(usuario))
    status_label.config(text="Sistema pronto.")

# Interface principal
root = tk.Tk()
root.title("Simulador de Carro Inteligente")
root.geometry("500x300")
root.resizable(False, False)

# Estilo visual
style = ttk.Style(root)
style.configure("TLabel", font=("Segoe UI", 14))
style.configure("TButton", font=("Segoe UI", 12))

# Painel
painel_frame = ttk.Frame(root, padding=20)
painel_frame.pack(expand=True)

painel_label = ttk.Label(painel_frame, text="Aguardando autenticação facial...", foreground="blue")
painel_label.pack(pady=10)

status_label = ttk.Label(painel_frame, text="Sistema iniciando...")
status_label.pack(pady=5)

# Simula o carro ligando após 1 segundo
root.after(1000, ligar_carro)

root.mainloop()
