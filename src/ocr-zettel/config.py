
# config.py
import os

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não foi encontrada. Crie um arquivo .env.")

WATCH_DIRECTORY = "/home/felipemarcelino/Google_Drive/onyx/TabUltraCPro/Notebooks/"

PDF_IMAGE_DPI = 200
PDF_ENABLE_CROP = True
PDF_CROP_BOX = (100, 150, 700, 1000)
