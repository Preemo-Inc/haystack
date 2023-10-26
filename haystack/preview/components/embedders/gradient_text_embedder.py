import logging
from typing import Any, Dict, List, Optional

from haystack.preview import component, default_to_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install gradientai'") as gradientai_import:
    from gradientai import Gradient

logger = logging.getLogger(__name__)


@component
class GradientTextEmbedder:
    """
    A component for embedding strings using models hosted on Gradient AI (https://gradient.ai).
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

    def to_dict(self) -> dict:
        """
        Serialize the component to a Python dictionary.
        """
        return default_to_dict(self, workspace_id=self._gradient.workspace_id, model_name=self._model_name)

    def warm_up(self) -> None:
        """
        Load the embedding backend.
        """
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = self._gradient.get_embeddings_model(self._model_name)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        if not isinstance(text, str):
            raise TypeError(
                "GradientTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the GradientDocumentEmbedder."
            )

        if not hasattr(self, "_embedding_model"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        result = self._embedding_model.generate_embeddings(inputs=[{"input": text}])
        return {"embedding": result["embeddings"][0]["embedding"]}
