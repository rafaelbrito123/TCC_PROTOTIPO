import sys
import customtkinter as ctk
import tkinter.messagebox

# Captura o nome do usuário passado como argumento
usuario = sys.argv[1] if len(sys.argv) > 1 else "Desconhecido"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Simulador de Carro com Reconhecimento Facial")
app.geometry("500x400")

status_label = ctk.CTkLabel(app, text=f"Usuário reconhecido: {usuario}", font=ctk.CTkFont(size=18))
status_label.pack(pady=20)

painel = ctk.CTkFrame(app, width=300, height=200, corner_radius=15)
painel.pack(pady=10)

painel_status = ctk.CTkLabel(painel, text="Painel: LIGADO ✅", font=ctk.CTkFont(size=16))
painel_status.pack(pady=20)

def ligar_carro():
    tkinter.messagebox.showinfo("Carro", f"🚗 Carro ligado com sucesso, {usuario}!\nMotor funcionando!")
    painel_status.configure(text="Painel: MOTOR ATIVO 🔥")

ctk.CTkButton(app, text="🔑 Ligar Carro", command=ligar_carro).pack(pady=10)

app.mainloop()
