# logger_setup.py
import logging
import sys


def setup_logging():
    """Configura o sistema de logging para a aplicação."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("app.log"), # Salva os logs em um arquivo
            logging.StreamHandler(sys.stdout), # Mostra os logs no console
        ],
    )

    # Define o logger para a biblioteca da OpenAI para não poluir o log com muitos detalhes
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Sistema de logging configurado.")
    return logger
