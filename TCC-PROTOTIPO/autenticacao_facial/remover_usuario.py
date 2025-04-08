import os

# Caminho para os embeddings
EMBEDDINGS_DIR = r"D:\OneDrive\Documentos\TCC-PROTOTIPO\TCC-PROTOTIPO\autenticacao_facial\embeddings"

# Lista os arquivos de usuários
usuarios = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]

if not usuarios:
    print("❌ Nenhum usuário cadastrado.")
    exit()

print("Usuários cadastrados:")
for i, arquivo in enumerate(usuarios):
    print(f"{i+1}. {arquivo.replace('.npy', '')}")

try:
    escolha = int(input("\nDigite o número do usuário que deseja remover: ")) - 1
    if escolha < 0 or escolha >= len(usuarios):
        print("❌ Opção inválida.")
    else:
        caminho = os.path.join(EMBEDDINGS_DIR, usuarios[escolha])
        os.remove(caminho)
        print(f"✅ Usuário '{usuarios[escolha].replace('.npy', '')}' removido com sucesso.")
except ValueError:
    print("❌ Entrada inválida. Digite um número.")
except Exception as e:
    print("❌ Erro ao remover o arquivo:", e)
