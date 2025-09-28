# gpt_vision_client.py
import base64
import io
import logging
from typing import List

import config
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

# Inicializa o cliente da OpenAI uma vez
try:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Falha ao inicializar o cliente da OpenAI. Verifique sua API Key. Erro: {e}")
    client = None

def get_markdown_from_vision(local_ocr_text: str, images: List[Image.Image]) -> str:
    """Envia o texto do OCR local e as imagens do PDF para o GPT-4 Vision e pede
    uma versão corrigida e formatada em Markdown.

    Args:
        local_ocr_text (str): O texto preliminar extraído pelo OCR local.
        images (List[Image.Image]): A lista de imagens das páginas do PDF.

    Returns:
        str: O conteúdo em Markdown retornado pela API.

    """
    if not client:
        logger.error("Cliente da OpenAI não inicializado. Abortando requisição.")
        return "Erro: Cliente da API não configurado."

    logger.info("Preparando requisição para a API do GPT-4 Vision...")

    # Converte imagens para base64
    base64_images = []
    for img in images:
        buffered = io.BytesIO()
        # Salva a imagem em formato PNG para manter a qualidade
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(f"data:image/png;base64,{img_str}")

    # Monta o prompt
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A seguir estão as páginas de uma nota que escaneei. "
                        "Eu usei um OCR local que extraiu o seguinte texto preliminar:\n\n"
                        f"--- INÍCIO DO TEXTO DO OCR LOCAL ---\n{local_ocr_text}\n--- FIM DO TEXTO DO OCR LOCAL ---\n\n"
                        "Sua tarefa é analisar as imagens das páginas com atenção, ignorar o texto do OCR local se ele estiver incorreto, "
                        "e fornecer uma transcrição completa e precisa do conteúdo. "
                        "Formate toda a sua resposta final em Markdown, preservando a estrutura como títulos, listas, negrito, etc. "
                        "Seja fiel ao conteúdo original da nota."
                        "Não precisa incluir que houve transcrição."
                        "Não precisa incluir ``` entre o texto, apenas as tags e códigos com o markdown."
                    ),
                },
                # Adiciona cada imagem à mensagem
                *[
                    {"type": "image_url", "image_url": {"url": b64_img}}
                    for b64_img in base64_images
                ],
            ],
        },
    ]

    try:
        logger.info("Enviando requisição para a API. Isso pode demorar...")
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=prompt_messages,
            max_tokens=4096,  # Aumente se suas notas forem muito longas
        )

        markdown_content = response.choices[0].message.content
        logger.info("Resposta recebida com sucesso da API.")
        return markdown_content
    except Exception as e:
        logger.error(f"Erro ao chamar a API do GPT-4 Vision: {e}", exc_info=True)
        return f"## Erro na API\n\nOcorreu um erro ao processar a nota: {e}"
