# pdf_processor.py
import logging
from typing import List

import fitz  # PyMuPDF
from config import PDF_IMAGE_DPI, PDF_ENABLE_CROP, PDF_CROP_BOX
from PIL import Image

logger = logging.getLogger(__name__)

def process_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Abre um arquivo PDF, converte cada página para uma imagem em escala de cinza,
    aplica um corte (se ativado) e retorna uma lista de objetos de imagem (Pillow).

    Args:
        pdf_path (str): O caminho para o arquivo PDF.

    Returns:
        List[Image.Image]: Uma lista de imagens, uma para cada página do PDF.

    """
    images = []
    try:
        logger.info(f"Iniciando pré-processamento do PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Converte a página para uma imagem (pixmap) em escala de cinza (csGRAY)
            zoom = PDF_IMAGE_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

            # Converte o pixmap para um objeto de imagem do Pillow
            img = Image.frombytes("G", [pix.width, pix.height], pix.samples)

            # --- LÓGICA DE CORTE ADICIONADA ---
            # Verifica se o corte está ativado na configuração
            if PDF_ENABLE_CROP:
                # Valida se a caixa de corte foi definida corretamente
                if not PDF_CROP_BOX or len(PDF_CROP_BOX) != 4:
                    logger.warning("Corte (crop) está ativado, mas PDF_CROP_BOX é inválido. Pulando corte.")
                else:
                    try:
                        logger.info(f"Aplicando corte na página {page_num + 1} com as coordenadas: {PDF_CROP_BOX}")
                        img = img.crop(PDF_CROP_BOX)
                    except Exception as e:
                        logger.error(f"Falha ao aplicar o corte na imagem da página {page_num + 1}: {e}", exc_info=True)

            images.append(img)

        doc.close()
        logger.info(f"PDF '{pdf_path}' processado com sucesso. {len(images)} páginas convertidas.")
        return images
    except Exception as e:
        logger.error(f"Falha ao processar o PDF '{pdf_path}': {e}", exc_info=True)
        return []
