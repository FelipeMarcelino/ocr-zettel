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

    logger.info("Preparando requisição para a API do GPT-5...")

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
                            f"""
                            Sua tarefa tem duas fases: transcrever, formatar

                            **FASE 1: TRANSCRIÇÃO**
                            Analise as imagens da nota manuscrita com atenção máxima aos detalhes. Ignore completamente o texto preliminar do OCR local, pois ele pode conter erros. Sua meta é obter uma transcrição 100% fiel e precisa do conteúdo das imagens.

                            **FASE 2: FORMATAÇÃO**
                            Depois de transcrever mentalmente o texto, formate sua resposta final usando estritamente a sintaxe Markdown.
                            - Preserve todos os títulos (ex: linhas começando com #, ##), com exeção de tags que explico
                              abaixo.
                            - Preserve listas com marcadores (-, *) ou números (1., 2.).
                            - Preserve qualquer formatação de **negrito** ou *itálico*.
                            - Preserve quebras de linha e parágrafos.
                            - Onde estiver escrito tag + uma palavra tipo assim -> tag: carreira, transforme em tag
                              markdown desta forma #carreira, sem espaço entre o hashtag e a palavra
                            - Seja criativo em organizar a nota, utilize tabelas, callouts, listas, negrito, itálico. 
                            - Para equações utilize latex no markdown.
                            - Caso haja diagramas tente formatar utilizando mermaid flowchart no markdown.
                            


                            **REGRA DE SAÍDA OBRIGATÓRIA:**
                            Sua resposta deve ser **APENAS** o texto Markdown transcrito. Não inclua NENHUMA palavra, frase ou comentário introdutório como "Aqui está a transcrição:". Não envolva a resposta final em blocos de código ```markdown. A saída deve ser o conteúdo puro, pronto para ser salvo em um arquivo .md.

                            --- INÍCIO DO TEXTO DO OCR LOCAL (APENAS PARA CONTEXTO, IGNORE SE ESTIVER ERRADO) ---
                            {local_ocr_text}
                            --- FIM DO TEXTO DO OCR LOCAL ---"""
                    ),
                },
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
            model="gpt-5",
            messages=prompt_messages,
            max_completion_tokens=4096,  # Aumente se suas notas forem muito longas
        )

        markdown_content = response.choices[0].message.content
        logger.info("Resposta recebida com sucesso da API.")
        return markdown_content
    except Exception as e:
        logger.error(f"Erro ao chamar a API do GPT-4 Vision: {e}", exc_info=True)
        return f"## Erro na API\n\nOcorreu um erro ao processar a nota: {e}"
