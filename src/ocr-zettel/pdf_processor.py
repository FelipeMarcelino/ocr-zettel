# pdf_ocr_pipeline.py
import logging
import os
from typing import List

import config
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --- Inicialização do motor TrOCR ---

# --- Funções de pré-processamento ---
def preprocess_page(pil_img: Image.Image, target_width=1600) -> Image.Image:
    """Pré-processa a página: converte para grayscale, binariza e redimensiona."""
    img = np.array(pil_img.convert("L"))
    scale = target_width / img.shape[1]
    img = cv2.resize(img, (target_width, int(img.shape[0] * scale)))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 10)
    return Image.fromarray(img)

def segment_blocks_and_lines(pil_img: Image.Image,
                             min_block_height=40,
                             min_line_height=20) -> List[Image.Image]:
    """Segmenta a imagem em linhas de texto, preservando a ordem natural."""
    img = np.array(pil_img.convert("L"))
    _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Detecta blocos
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 50))
    dilated_blocks = cv2.dilate(bin_img, kernel_block, iterations=1)
    contours, _ = cv2.findContours(dilated_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_block_height: continue
        block_img = pil_img.crop((x, y, x + w, y + h))
        blocks.append((y, block_img))
    blocks = sorted(blocks, key=lambda x: x[0])

    lines = []
    for _, block in blocks:
        block_arr = np.array(block.convert("L"))
        _, block_bin = cv2.threshold(block_arr, 200, 255, cv2.THRESH_BINARY_INV)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        dilated_lines = cv2.dilate(block_bin, kernel_line, iterations=1)
        line_contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in line_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < min_line_height: continue
            line_img = block.crop((x, y, x + w, y + h))
            lines.append((y, line_img))
    lines = [img for y, img in sorted(lines, key=lambda x: x[0])]
    return lines

# --- Função para processar PDF ---
def process_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    images = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        logger.info(f"Processando PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        render_zoom = config.OCR_RESOLUTION_DPI / 72.0
        mat = fitz.Matrix(render_zoom, render_zoom)
        coord_scale_factor = config.OCR_RESOLUTION_DPI / config.SCREEN_ASSUMED_DPI

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGBA" if pix.alpha else "RGB",
                                  [pix.width, pix.height],
                                  pix.samples)
            if pix.alpha:
                img = img.convert("RGB")

            # Crop
            if config.PDF_ENABLE_CROP and config.PDF_CROP_BOX and len(config.PDF_CROP_BOX) == 4:
                try:
                    scaled_crop_box = tuple(int(c * coord_scale_factor) for c in config.PDF_CROP_BOX)
                    img = img.crop(scaled_crop_box)
                except Exception as e:
                    logger.error(f"Erro ao aplicar crop: {e}", exc_info=True)

            if config.DEBUG_SAVE_IMAGES:
                debug_filename = f"debug_{base_filename}_page_{page_num+1}.png"
                img.save(debug_filename)
                logger.info(f"Imagem de debug salva em: {os.path.abspath(debug_filename)}")

            images.append(img)
        doc.close()
        return images
    except Exception as e:
        logger.error(f"Falha ao processar PDF: {e}", exc_info=True)
        return []

