import os

from langchain_ollama import OllamaEmbeddings, ChatOllama

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from dotenv import load_dotenv

load_dotenv()

ollama_url = os.getenv("OLLAMA")
embed_model = os.getenv("EMBED_MODEL")
chat_model = os.getenv("CHAT_MODEL")

llm = ChatOllama(
    model=chat_model,
    base_url=ollama_url,
    temperature=0
)

embeddings = OllamaEmbeddings(
    model=embed_model,
    base_url=ollama_url
)

eval_llm = LangchainLLMWrapper(llm)
eval_embeddings = LangchainEmbeddingsWrapper(embeddings)