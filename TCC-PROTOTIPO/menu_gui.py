import os
import subprocess
import customtkinter as ctk
import tkinter.messagebox

# Diretório onde estão salvos os embeddings .npy
EMBEDDINGS_DIR = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\embeddings"

# Caminho dos scripts
PYTHON_EXEC = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\.venv\Scripts\python.exe"
SCRIPT_CADASTRO = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\cadastro_usuarios_avancado.py"
SCRIPT_AUTENTICACAO = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\autenticacao_ao_vivo.py"

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
    subprocess.run([PYTHON_EXEC, SCRIPT_CADASTRO])

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
