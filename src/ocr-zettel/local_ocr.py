# local_ocr.py
import logging
from typing import List

import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --- Inicialização do Motor EasyOCR ---
# O EasyOCR é inicializado uma vez. A primeira execução irá baixar os modelos
# para o idioma especificado (neste caso, português).
try:
    logger.info("Inicializando o motor EasyOCR para o idioma 'pt' (Português)...")
    # 'gpu=False' força o uso da CPU se você não tiver uma GPU configurada.
    # Se tiver uma GPU NVIDIA com CUDA, pode tentar 'gpu=True'.
    ocr_reader = easyocr.Reader(["pt"], gpu=False)
    logger.info("Motor EasyOCR inicializado com sucesso.")
except Exception as e:
    logger.error(f"Falha ao inicializar o motor EasyOCR: {e}", exc_info=True)
    ocr_reader = None

def extract_text_with_easyocr(images: List[Image.Image]) -> str:
    """Usa a biblioteca EasyOCR para extrair texto de uma lista de imagens.

    Args:
        images (List[Image.Image]): Lista de imagens pré-processadas do PDF.

    Returns:
        str: O texto extraído, com páginas separadas por um divisor.

    """
    if not ocr_reader:
        logger.error("Motor EasyOCR não está disponível. Pulando extração de texto local.")
        return "[ERRO: OCR LOCAL NÃO INICIALIZADO]"

    all_pages_text = []
    logger.info(f"Iniciando extração de texto com EasyOCR para {len(images)} página(s)...")

    try:
        for i, img in enumerate(images):
            # Converte a imagem do Pillow para um array NumPy
            img_np = np.array(img)

            # O EasyOCR lê o texto da imagem. 'detail=0' retorna apenas o texto.
            result = ocr_reader.readtext(img_np, detail=0, paragraph=True)

            if result:
                page_text = "\n".join(result)
                all_pages_text.append(page_text)
                logger.info(f"Texto extraído da página {i+1}.")
            else:
                logger.info(f"Nenhum texto detectado na página {i+1} pelo EasyOCR.")
                all_pages_text.append(f"[Nenhum texto detectado na página {i+1}]")

        logger.info("Extração de texto com EasyOCR concluída.")
        return "\n\n---\n[Nova Página]\n---\n\n".join(all_pages_text)

    except Exception as e:
        logger.error(f"Erro durante a execução do EasyOCR: {e}", exc_info=True)
        return "[ERRO DURANTE A EXECUÇÃO DO OCR LOCAL]"
