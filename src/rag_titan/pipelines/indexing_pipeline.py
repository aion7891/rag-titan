"""
Offline indexing, shared by every RAG type:
    load files -> chunk -> embed -> store

A folder or a .zip in the file list is read as an OKF knowledge bundle: its
concepts become the documents (see okf/documents.py).
"""

from pathlib import Path

from rag_titan.loaders.loader_registry import load_document
from rag_titan.okf.documents import check_okf_options, okf_documents
from rag_titan.utils.logger import get_logger
from rag_titan.vector_stores.index_info import build_info, check_same_build

logger = get_logger(__name__)


class IndexingPipeline:
    """
    Prepares files so they can be searched.

    Args:
        chunker: a BaseChunker.
        embedder: a BaseEmbedder.
        store: a BaseStore.
        okf_include: for OKF bundles: "current" (leave out deprecated concepts)
            or "all".
        okf_min_trust: for OKF bundles: the lowest trust tier to index
            ("unverified" keeps all).
    """

    def __init__(self, chunker, embedder, store, okf_include="current",
                 okf_min_trust="unverified"):
        check_okf_options(okf_include, okf_min_trust)
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.okf_include, self.okf_min_trust = okf_include, okf_min_trust

    def run(self, file_paths, reset=True):
        """
        Index the files.

        Args:
            file_paths: list of file paths to index.
            reset: when True, clear the store before adding new chunks.
                When False, add to what is stored: a file indexed before is
                replaced, and chunk ids go on after the stored ones.

        Returns:
            The number of chunks stored.
        """
        documents = self.load_documents(file_paths)

        chunks = self.chunker.chunk(documents)
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created, nothing to index")
            return 0

        vectors = self.embedder.embed_many([chunk["text"] for chunk in chunks])
        dimensions = len(vectors[0])

        if reset:
            self.store.clear()
            first = 0
        else:
            first = self._make_room(documents, dimensions)

        for number, chunk in enumerate(chunks, start=first):
            chunk["id"] = f"chunk_{number}"     # unique in the whole index

        self.store.add(chunks, vectors)
        self.store.set_info(build_info(self.embedder, self.chunker, dimensions,
                                       first + len(chunks)))
        logger.info("Stored %d chunks in the vector store", self.store.count())

        return len(chunks)

    def _make_room(self, documents, dimensions):
        """
        Get the store ready to add more chunks; return the first new chunk number.

        Checks that the same embedder built the index (nothing is changed when
        it did not), then removes older chunks of files that are indexed again.
        """
        info = self.store.get_info()
        check_same_build(info, self.embedder.model, dimensions)

        for source in sorted({document["source"] for document in documents}):
            try:
                removed = self.store.delete_source(source)
            except NotImplementedError:
                logger.warning("%s cannot replace files; chunks of %s may be stored twice",
                               type(self.store).__name__, source)
                continue
            if removed:
                logger.info("Replaced %d old chunks of %s", removed, source)

        # stores without info (custom stores written before 0.10.0) count instead
        return info["next_chunk"] if info else self.store.count()

    def load_documents(self, file_paths):
        """Load every file (and every OKF bundle's concepts); skip the ones without text."""
        documents = []

        for path in file_paths:
            if is_bundle(path):
                documents += okf_documents(path, self.okf_include, self.okf_min_trust)
                continue
            document = load_document(path)

            if not document["text"].strip():
                logger.warning("Skipping empty document: %s", document["source"])
                continue

            word_count = len(document["text"].split())
            logger.info("Loaded %s (%d words)", document["source"], word_count)
            documents.append(document)

        return documents


def is_bundle(path):
    """True for a folder or a .zip file: they are read as OKF bundles."""
    path = Path(path)
    return path.is_dir() or (path.is_file() and path.suffix.lower() == ".zip")
