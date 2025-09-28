# local_ocr.py
import logging
from typing import List

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

# --- Inicialização do Motor TrOCR ---
# Carregamos o modelo e o processador uma única vez para evitar recarregá-los
# a cada arquivo processado. Isso pode consumir uma quantidade significativa de RAM.
try:
    logger.info("Inicializando o motor TrOCR da Microsoft (pode demorar na primeira vez)...")

    # Verifica se há uma GPU disponível e a define como dispositivo principal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"TrOCR usará o dispositivo: {device.type}")

    # O processador prepara a imagem para o modelo (redimensiona, normaliza, etc.)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # O modelo é a rede neural que faz a "leitura" da imagem
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

    logger.info("Motor TrOCR inicializado com sucesso.")

except Exception as e:
    logger.error(f"Falha ao inicializar o motor TrOCR. Verifique a instalação do PyTorch e transformers. Erro: {e}", exc_info=True)
    processor = None
    model = None
    device = "cpu"

def extract_text_with_trocr(images: List[Image.Image]) -> str:
    """Usa o modelo TrOCR da Microsoft para extrair texto manuscrito de imagens.
    """
    if not model or not processor:
        logger.error("Motor TrOCR não está disponível. Pulando extração de texto.")
        return "[ERRO: MOTOR TrOCR NÃO INICIALIZADO]"

    all_pages_text = []
    logger.info(f"Iniciando extração de texto com TrOCR para {len(images)} página(s)...")

    try:
        for i, img in enumerate(images):
            # O TrOCR espera imagens no formato RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Prepara a imagem usando o processador e envia para o dispositivo (CPU ou GPU)
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

            # Gera os IDs dos tokens a partir dos pixels da imagem
            generated_ids = model.generate(pixel_values)

            # Decodifica os IDs dos tokens para texto legível
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if generated_text:
                all_pages_text.append(generated_text)
                logger.info(f"Texto extraído da página {i+1}.")
            else:
                logger.info(f"Nenhum texto detectado na página {i+1} pelo TrOCR.")
                all_pages_text.append(f"[Nenhum texto detectado na página {i+1}]")

        logger.info("Extração de texto com TrOCR concluída.")
        return "\n\n---\n[Nova Página]\n---\n\n".join(all_pages_text)

    except Exception as e:
        logger.error(f"Erro durante a execução do TrOCR: {e}", exc_info=True)
        return "[ERRO DURANTE A EXECUÇÃO DO OCR LOCAL COM TrOCR]"
