# pdf_processor.py
import logging
import os
from typing import List

import fitz  # PyMuPDF

# Importa as configurações. As de binarização não são mais usadas.
from config import DEBUG_SAVE_IMAGES, OCR_RESOLUTION_DPI, PDF_CROP_BOX, PDF_ENABLE_CROP, SCREEN_ASSUMED_DPI
from PIL import Image

logger = logging.getLogger(__name__)


def process_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Abre um arquivo PDF, converte cada página para uma imagem de alta resolução em RGB
    e aplica um corte calibrado. Esta é a versão otimizada para o TrOCR.
    """
    images = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        logger.info(f"Iniciando pré-processamento do PDF para TrOCR: {pdf_path}")
        doc = fitz.open(pdf_path)

        render_zoom = OCR_RESOLUTION_DPI / 72.0
        mat = fitz.Matrix(render_zoom, render_zoom)

        coord_scale_factor = OCR_RESOLUTION_DPI / SCREEN_ASSUMED_DPI

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # --- MUDANÇA PRINCIPAL AQUI ---
            # Renderiza a imagem em RGB, não mais em Grayscale.
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            # Converte diretamente para uma imagem Pillow RGB.
            # A verificação de pix.n não é mais tão necessária, mas é uma boa prática.
            if pix.alpha: # Se tiver canal alfa (transparência), remove.
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if PDF_ENABLE_CROP:
                if not PDF_CROP_BOX or len(PDF_CROP_BOX) != 4:
                    logger.warning("Corte (crop) está ativado, mas PDF_CROP_BOX é inválido.")
                else:
                    try:
                        scaled_crop_box = (
                            int(PDF_CROP_BOX[0] * coord_scale_factor),
                            int(PDF_CROP_BOX[1] * coord_scale_factor),
                            int(PDF_CROP_BOX[2] * coord_scale_factor),
                            int(PDF_CROP_BOX[3] * coord_scale_factor),
                        )
                        logger.info(f"Aplicando corte com coordenadas escaladas: {scaled_crop_box}")
                        img = img.crop(scaled_crop_box)
                    except Exception as e:
                        logger.error(f"Falha ao aplicar o corte na página {page_num + 1}: {e}", exc_info=True)

            if DEBUG_SAVE_IMAGES:
                debug_filename = f"debug_{base_filename}_page_{page_num + 1}.png"
                img.save(debug_filename)
                logger.info(f"IMAGEM DE DEPURAÇÃO SALVA EM: {os.path.abspath(debug_filename)}")

            images.append(img)

        doc.close()
        logger.info(f"PDF '{pdf_path}' processado com sucesso. {len(images)} páginas convertidas.")
        return images
    except Exception as e:
        logger.error(f"Falha ao processar o PDF '{pdf_path}': {e}", exc_info=True)
        return []
