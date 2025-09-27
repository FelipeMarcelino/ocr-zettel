
import logging
import time

import config
from file_handler import PDFChangeHandler
from logger_setup import setup_logging
from watchdog.observers import Observer

# Configura o logging antes de qualquer outra coisa
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Função principal para iniciar o monitoramento."""
    logger.info("Iniciando o serviço de monitoramento de notas...")
    logger.info(f"Monitorando o diretório: {config.WATCH_DIRECTORY}")

    if not os.path.isdir(config.WATCH_DIRECTORY):
        logger.error(f"O diretório '{config.WATCH_DIRECTORY}' não existe. Verifique o arquivo config.py.")
        return

    # Cria o manipulador de eventos
    event_handler = PDFChangeHandler()

    # Cria e configura o observador
    observer = Observer()
    observer.schedule(event_handler, config.WATCH_DIRECTORY, recursive=True)

    # Inicia o observador em uma thread separada
    observer.start()
    logger.info("Observador iniciado. Pressione CTRL+C para parar.")

    try:
        # Mantém o script principal rodando para que o observador continue ativo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Sinal de interrupção recebido. Parando o observador...")
        observer.stop()

    # Aguarda a thread do observador terminar
    observer.join()
    logger.info("Observador parado. O programa foi encerrado.")

if __name__ == "__main__":
    # Verifica a existência da chave de API antes de rodar
    import os
    if not config.OPENAI_API_KEY:
        logger.error("A chave da API da OpenAI não foi configurada. Encerrando.")
    else:
        main()
