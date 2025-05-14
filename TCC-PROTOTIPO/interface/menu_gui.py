import os
import subprocess
import customtkinter as ctk
import tkinter.messagebox
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import PYTHON_EXEC, SCRIPT_AUTENTICACAO, SCRIPT_CADASTRO, EMBEDDINGS_DIR

# Caminho dos scripts


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Sistema de Autenticação Facial")
app.geometry("500x400")

def listar_usuarios():
    if not os.path.exists(EMBEDDINGS_DIR):
        tkinter.messagebox.showerror("Erro", "Pasta de embeddings não encontrada!")
        return
    arquivos = [f[:-4] for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]
    if arquivos:
        lista = "\n".join(arquivos)
        tkinter.messagebox.showinfo("Usuários Cadastrados", lista)
    else:
        tkinter.messagebox.showinfo("Usuários Cadastrados", "Nenhum usuário cadastrado.")

def remover_usuario():
    listar_usuarios()
    nome = tkinter.simpledialog.askstring("Remover Usuário", "Digite o nome do usuário:")
    if nome:
        nome = nome.strip().lower().replace(" ", "_")
        caminho = os.path.join(EMBEDDINGS_DIR, f"{nome}.npy")
        if os.path.exists(caminho):
            os.remove(caminho)
            tkinter.messagebox.showinfo("Sucesso", f"Usuário '{nome}' removido.")
        else:
            tkinter.messagebox.showerror("Erro", f"Usuário '{nome}' não encontrado.")

def cadastrar_usuario():
    nome = tkinter.simpledialog.askstring("Cadastro de Usuário", "Digite o nome do novo usuário:")
    if nome:
        nome = nome.strip().lower().replace(" ", "_")
        subprocess.run([PYTHON_EXEC, SCRIPT_CADASTRO, nome])

def autenticar_usuario():
    subprocess.run([PYTHON_EXEC, SCRIPT_AUTENTICACAO])

# Layout dos botões
ctk.CTkLabel(app, text="🔐 Menu de Autenticação Facial", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

ctk.CTkButton(app, text="➕ Cadastrar Novo Usuário", command=cadastrar_usuario).pack(pady=10)
ctk.CTkButton(app, text="📷 Autenticar Usuário ao Vivo", command=autenticar_usuario).pack(pady=10)
ctk.CTkButton(app, text="🗑️ Remover Usuário", command=remover_usuario).pack(pady=10)
ctk.CTkButton(app, text="📂 Listar Usuários", command=listar_usuarios).pack(pady=10)
ctk.CTkButton(app, text="🚪 Sair", command=app.destroy, fg_color="red").pack(pady=20)

app.mainloop()
