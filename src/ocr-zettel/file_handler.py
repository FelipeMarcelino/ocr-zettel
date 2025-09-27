# file_handler.py
import logging
import os
import time

import gpt_vision_client
import local_ocr
import pdf_processor
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class PDFChangeHandler(FileSystemEventHandler):
    """Manipulador de eventos que reage à criação e modificação de arquivos PDF.
    """

    def __init__(self):
        super().__init__()
        # Armazena o timestamp do último processamento por arquivo para evitar duplicações
        self.processed_files = {}

    def _should_process(self, path: str) -> bool:
        """Verifica se um arquivo deve ser processado."""
        # Ignora diretórios
        if os.path.isdir(path):
            return False
        # Processa apenas arquivos .pdf
        if not path.lower().endswith(".pdf"):
            return False
        # Ignora arquivos temporários (comuns em syncs)
        if os.path.basename(path).startswith("~") or os.path.basename(path).startswith("."):
            return False

        # Lógica para evitar processamento duplicado rápido
        # (às vezes, um único save dispara múltiplos eventos)
        now = time.time()
        last_processed_time = self.processed_files.get(path, 0)
        if now - last_processed_time < 5: # Ignora se processado nos últimos 5 segundos
            return False

        return True

    def _process_file(self, pdf_path: str):
        """Orquestra o fluxo de processamento para um único arquivo PDF."""
        if not self._should_process(pdf_path):
            return

        logger.info(f"Evento detectado para o arquivo: {pdf_path}. Iniciando processamento.")
        self.processed_files[pdf_path] = time.time() # Marca como processado

        try:
            # 1. Pré-processar o PDF para obter imagens
            images = pdf_processor.process_pdf_to_images(pdf_path)
            if not images:
                logger.warning(f"Nenhuma imagem foi extraída de '{pdf_path}'. Abortando.")
                return

            # 2. Extrair texto com OCR local
            local_text = local_ocr.extract_text_with_onnx(images)

            # 3. Chamar a API GPT-4 Vision para obter o Markdown
            markdown_result = gpt_vision_client.get_markdown_from_vision(local_text, images)

            # 4. Salvar o resultado em um arquivo .md
            markdown_path = os.path.splitext(pdf_path)[0] + ".md"
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_result)

            logger.info(f"Arquivo Markdown salvo com sucesso em: {markdown_path}")

        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado no fluxo de processamento para '{pdf_path}': {e}", exc_info=True)


    def on_created(self, event):
        """Chamado quando um arquivo ou diretório é criado."""
        self._process_file(event.src_path)

    def on_modified(self, event):
        """Chamado quando um arquivo ou diretório é modificado."""
        self._process_file(event.src_path)
