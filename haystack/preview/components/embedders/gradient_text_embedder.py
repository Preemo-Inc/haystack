import logging
import os
from typing import List, Optional

from haystack.preview import component
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
        self, *, access_token: Optional[str] = None, workspace_id: Optional[str] = None, host: Optional[str] = None
    ) -> None:
        self._host = host

        if access_token is None:
            try:
                access_token = os.environ["GRADIENT_ACCESS_TOKEN"]
            except KeyError as e:
                raise ValueError(
                    "GradientTextEmbedder expects an access token. "
                    "Set the GRADIENT_ACCESS_TOKEN environment variable or pass it explicitly."
                ) from e

        self._gradient = Gradient(access_token=access_token, host=host, workspace_id=workspace_id)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        return {"embedding": [0.0, 0.0, 0.0]}
