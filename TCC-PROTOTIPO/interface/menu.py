import os
import subprocess

EMBEDDINGS_DIR = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\embeddings"
SCRIPT_CADASTRO = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\cadastro_usuarios_avancado.py"
SCRIPT_AUTENTICACAO = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\autenticacao_ao_vivo.py"
VENV_PYTHON = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\.venv\Scripts\python.exe"
  # Caminho do Python da venv

def listar_usuarios():
    print("\n📂 Usuários cadastrados:")
    arquivos = [f[:-4] for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]
    if arquivos:
        for usuario in arquivos:
            print(f"  - {usuario}")
    else:
        print("  (nenhum usuário cadastrado)")

def remover_usuario():
    listar_usuarios()
    nome = input("\nDigite o nome do usuário para remover: ").strip().lower().replace(" ", "_")
    caminho = os.path.join(EMBEDDINGS_DIR, f"{nome}.npy")
    if os.path.exists(caminho):
        os.remove(caminho)
        print(f"✅ Usuário '{nome}' removido com sucesso.")
    else:
        print(f"❌ Usuário '{nome}' não encontrado.")

def main():
    while True:
        print("\n===== MENU DE AUTENTICAÇÃO FACIAL =====")
        print("1. Cadastrar novo usuário")
        print("2. Autenticar usuário ao vivo")
        print("3. Remover usuário cadastrado")
        print("4. Listar usuários cadastrados")
        print("5. Sair")
        escolha = input("Escolha uma opção: ")

        if escolha == '1':
            subprocess.run([VENV_PYTHON, SCRIPT_CADASTRO])
        elif escolha == '2':
            subprocess.run([VENV_PYTHON, SCRIPT_AUTENTICACAO])
        elif escolha == '3':
            remover_usuario()
        elif escolha == '4':
            listar_usuarios()
        elif escolha == '5':
            print("Encerrando...")
            break
        else:
            print("❗ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
