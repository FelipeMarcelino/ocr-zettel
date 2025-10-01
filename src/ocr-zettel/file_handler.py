# file_handler.py
import logging
import os
import time

import gpt_vision_client
import local_ocr
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

import logging

# file_handler.py


logger = logging.getLogger(__name__)

class PDFChangeHandler(FileSystemEventHandler):
    """Manipulador de eventos que reage à criação e modificação de arquivos PDF,
    com uma verificação de estabilidade para lidar com sincronização de nuvem.
    """

    def __init__(self):
        super().__init__()
        self.processed_files = {}

    def _should_process(self, path: str) -> bool:
        """Verifica se um arquivo deve ser processado com base no nome e no tempo."""
        if os.path.isdir(path) or not path.lower().endswith(".pdf") or os.path.basename(path).startswith("~"):
            return False

        # Evita processamento duplicado muito rápido (debounce)
        now = time.time()
        last_processed_time = self.processed_files.get(path, 0)
        if now - last_processed_time < 10: # Aumentado para 10 segundos
            return False

        return True

    def _is_file_stable(self, file_path: str) -> bool:
        """Verifica se o arquivo parou de ser modificado, aguardando até que seu tamanho
        permaneça o mesmo por alguns segundos.
        """
        if not os.path.exists(file_path):
            return False

        logger.info(f"Verificando estabilidade do arquivo: {file_path}")
        try:
            last_size = -1
            stable_count = 0
            # O arquivo precisa ter o mesmo tamanho por 3 segundos consecutivos
            stability_threshold = 3

            while stable_count < stability_threshold:
                if not os.path.exists(file_path):
                    logger.warning(f"Arquivo {file_path} desapareceu durante verificação. Cancelando.")
                    return False

                current_size = os.path.getsize(file_path)

                if current_size == 0:
                    logger.debug(f"Arquivo {file_path} ainda está com 0 bytes. Aguardando...")
                    stable_count = 0 # Reinicia se o arquivo ainda estiver vazio
                    time.sleep(1)
                    continue

                if current_size == last_size:
                    stable_count += 1
                else:
                    last_size = current_size
                    stable_count = 0  # Reinicia a contagem se o tamanho mudar

                time.sleep(1)

            logger.info(f"Arquivo está estável com {last_size} bytes.")
            return True

        except FileNotFoundError:
            logger.warning(f"Arquivo {file_path} não encontrado durante verificação. Pode ter sido temporário.")
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar estabilidade do arquivo {file_path}: {e}")
            return False


    def _process_file(self, pdf_path: str):
        """Orquestra o fluxo de processamento para um único arquivo PDF."""
        if not self._should_process(pdf_path):
            return

        # --- NOVA LÓGICA DE ESTABILIDADE ---
        if not self._is_file_stable(pdf_path):
            logger.warning(f"Processamento de {pdf_path} cancelado por instabilidade do arquivo.")
            return

        logger.info(f"Arquivo estável. Iniciando processamento completo de: {pdf_path}")
        self.processed_files[pdf_path] = time.time() # Marca como processado

        try:
            # 1. Extrair texto com OCR local
            local_text, images = local_ocr.extract_text_from_pdf(pdf_path)

            # 2. Chamar a API GPT-4o para obter o Markdown
            markdown_result = gpt_vision_client.get_markdown_from_vision(local_text, images)

            # 3. Salvar o resultado em um arquivo .md
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
