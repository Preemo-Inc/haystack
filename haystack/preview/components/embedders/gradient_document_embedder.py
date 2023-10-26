import logging
from typing import List, Optional, Dict, Any

from haystack.preview import component, Document, default_to_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install gradientai'") as gradientai_import:
    from gradientai import Gradient

logger = logging.getLogger(__name__)


@component
class GradientDocumentEmbedder:
    """
    A component for computing Document embeddings using Gradient AI API..
    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(
        self,
        *,
        model_name: str = "bge-large",
        access_token: Optional[str] = None,
        workspace_id: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        self._host = host
        self._model_name = model_name

        self._gradient = Gradient(access_token=access_token, host=host, workspace_id=workspace_id)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self._model_name}

    def warm_up(self) -> None:
        """
        Load the embedding backend.
        """
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = self._gradient.get_embeddings_model(slug=self._model_name)

    def to_dict(self) -> dict:
        """
        Serialize the component to a Python dictionary.
        """
        return default_to_dict(self, workspace_id=self._gradient.workspace_id, model_name=self._model_name)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "GradientDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the GradientTextEmbedder."
            )

        if not hasattr(self, "_embedding_model"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        result = self._embedding_model.generate_embeddings(inputs=[{"input": doc.text} for doc in documents])
        for doc, embedding in zip(documents, result.embeddings):
            doc.embedding = embedding.embedding

        return {"documents": documents}
