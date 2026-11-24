"""
Embedding model for any server that speaks the OpenAI Embeddings API: OpenAI,
Gemini, OpenRouter, LM Studio, vLLM, Ollama's /v1, and more.

Only the standard library is used. The key is sent as "Authorization: Bearer
<key>" and never shown in errors.

Example:
    from rag_titan.embeddings import OpenAICompatibleEmbedder

    embedder = OpenAICompatibleEmbedder("nomic-embed-text", "http://localhost:11434/v1")
    vector = embedder.embed("Staff get twenty leave days.")
"""

import numpy as np

from rag_titan.embeddings.base_embedder import BaseEmbedder
from rag_titan.llm.http_json import post_json
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAICompatibleEmbedder(BaseEmbedder):
    """
    Creates embeddings with POST {base_url}/embeddings.

    Args:
        model: the embedding model name, for example "text-embedding-3-small".
        base_url: the API address up to /v1, for example "https://api.openai.com/v1".
        api_key: the key; None or "" for local servers that need none.
        service: the provider's name, used in error messages.
        key_name: where the key comes from, named in the "no key" error;
            None means the server needs no key.
        batch_size: how many texts to send in one request.
    """

    def __init__(self, model, base_url, api_key=None, service="The API", key_name=None,
                 batch_size=64):
        super().__init__(model)
        if not model:
            raise ValueError(f"{service} needs an embedding model name: set EMBED_MODEL")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {base_url!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a whole number above 0, got {batch_size!r}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.service, self.key_name = service, key_name
        self.batch_size = batch_size

    def __repr__(self):
        # never show the key, even when someone prints the object
        return f"OpenAICompatibleEmbedder(model={self.model!r}, base_url={self.base_url!r})"

    def embed_many(self, texts):
        """Return one numpy vector per text, in the same order."""
        if self.key_name and not self.api_key:
            raise ValueError(f"{self.service} needs an API key: set the {self.key_name} "
                             "environment variable (or .env)")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
            logger.debug("Embedded %d / %d texts", len(vectors), len(texts))

        # a store can only compare vectors of one size
        sizes = {len(vector) for vector in vectors}
        if len(sizes) > 1:
            raise RuntimeError(f"{self.service} sent vectors of different sizes {sorted(sizes)} "
                               f"(model '{self.model}')")
        return vectors

    def _embed_batch(self, batch):
        """Send one batch and return its vectors in the order of the texts."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            reply = post_json(f"{self.base_url}/embeddings",
                              {"model": self.model, "input": list(batch)}, headers,
                              service=self.service)
        except RuntimeError as error:
            raise RuntimeError(
                f"{self.service} embedding failed for model '{self.model}': {error}") from None

        items = reply.get("data") if isinstance(reply, dict) else None
        # one vector per text, or the chunks and vectors would get out of step
        if not isinstance(items, list) or len(items) != len(batch):
            raise RuntimeError(f"{self.service} sent {len(items or [])} embeddings for "
                               f"{len(batch)} texts (model '{self.model}')")
        # the API says each item's "index"; sort by it rather than trust the order
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        if [item.get("index", i) for i, item in enumerate(ordered)] != list(range(len(batch))):
            raise RuntimeError(f"{self.service} sent embeddings with wrong indexes "
                               f"(model '{self.model}')")
        if not all(isinstance(item.get("embedding"), list) and item["embedding"]
                   for item in ordered):
            raise RuntimeError(f"{self.service} sent an item without an embedding list "
                               f"(model '{self.model}')")
        return [np.array(item["embedding"], dtype=np.float32) for item in ordered]
