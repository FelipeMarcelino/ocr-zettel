# local_ocr.py
import logging
from typing import List

import config
import torch
from pdf_processor import (preprocess_page, process_pdf_to_images,
                           segment_blocks_and_lines)
from PIL import Image
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


spell = SpellChecker(language="pt")

# --- Inicialização do Motor TrOCR ---
# Carregamos o modelo e o processador uma única vez para evitar recarregá-los
# a cada arquivo processado. Isso pode consumir uma quantidade significativa de RAM.
try:
    logger.info("Inicializando o motor TrOCR da Microsoft (pode demorar na primeira vez)...")

    # Verifica se há uma GPU disponível e a define como dispositivo principal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"TrOCR usará o dispositivo: {device.type}")

    # O processador prepara a imagem para o modelo (redimensiona, normaliza, etc.)
    processor = TrOCRProcessor.from_pretrained(config.FINAL_MODEL_PATH)

    # O modelo é a rede neural que faz a "leitura" da imagem
    model = VisionEncoderDecoderModel.from_pretrained(config.FINAL_MODEL_PATH).to(device)

    logger.info("Motor TrOCR inicializado com sucesso.")

except Exception as e:
    logger.error(f"Falha ao inicializar o motor TrOCR. Verifique a instalação do PyTorch e transformers. Erro: {e}", exc_info=True)
    processor = None
    model = None
    device = "cpu"


def clean_text(text: str) -> str:
    """Corrige palavras do texto usando spell checker em português."""
    words = text.split()
    corrected = [spell.correction(w) if w not in spell else w for w in words]
    corrected_filtered = " ".join([s for s in corrected  if s is not None])
    return " ".join(corrected_filtered)

def extract_text_with_trocr(images: List[Image.Image], page_number: int) -> str:
    """Usa o modelo TrOCR da Microsoft para extrair texto manuscrito de imagens.
    """
    if not model or not processor:
        logger.error("Motor TrOCR não está disponível. Pulando extração de texto.")
        return "[ERRO: MOTOR TrOCR NÃO INICIALIZADO]"

    all_lines_text = []
    logger.info(f"Iniciando extração de texto com TrOCR para {len(images)} página(s)...")

    try:
        for i, img in enumerate(images):
            # O TrOCR espera imagens no formato RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Prepara a imagem usando o processador e envia para o dispositivo (CPU ou GPU)
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

            # Gera os IDs dos tokens a partir dos pixels da imagem
            generated_ids = model.generate(pixel_values)i

            # Decodifica os IDs dos tokens para texto legível
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if generated_text:
                cleaned_text = clean_text(generated_text)
                all_lines_text.append(cleaned_text)
                logger.info(f"Texto extraído da linha {i+1}, página {pg_number}.")
            else:
                logger.info(f"Nenhum texto detectado na página linha {i+1}, página {pg_number} pelo TrOCR.")
                all_lines_text.append(f"[Nenhum texto detectado na linha {i+1}, página {pg_number}]")

        logger.info("Extração de texto com TrOCR concluída.")
        return "\n".join(all_lines_text)

    except Exception as e:
        logger.error(f"Erro durante a execução do TrOCR: {e}", exc_info=True)
        return "[ERRO DURANTE A EXECUÇÃO DO OCR LOCAL COM TrOCR]"


def extract_text_from_pdf(pdf_path: str) -> str:
    images = process_pdf_to_images(pdf_path)
    if not images:
        return "[ERRO: Nenhuma imagem extraída do PDF]"

    all_text = []
    for page_num, page_img in enumerate(images):
        preprocessed_img = preprocess_page(page_img)
        lines_img = segment_blocks_and_lines(preprocessed_img)
        lines_txt = extract_text_with_trocr(lines_img, pag_num)
        all_text.append(lines_txt)

    return ("\n\n---\n[Nova Página]\n---\n\n".join(all_text), images)
