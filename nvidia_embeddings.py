import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class NvidiaNIMEmbeddings(Embeddings):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is required for NVIDIA embeddings.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if isinstance(t, str) else str(t) for t in texts[i : i + self.batch_size]]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                extra_body={"input_type": input_type},
            )
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], input_type="query")[0]
