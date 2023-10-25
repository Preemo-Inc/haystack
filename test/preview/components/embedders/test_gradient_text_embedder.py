import os
import pytest
from haystack.preview.components.embedders.gradient_text_embedder import GradientTextEmbedder


class TestGradientTextEmbedder:
    @pytest.mark.unit
    def test_init_from_env(self):
        access_token = "access_token"
        workspace_id = "workspace_id"

        os.environ["GRADIENT_ACCESS_TOKEN"] = access_token
        os.environ["GRADIENT_WORKSPACE_ID"] = workspace_id

        embedder = GradientTextEmbedder()
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_without_access_token(self):
        workspace_id = "workspace_id"
        os.environ.pop("GRADIENT_ACCESS_TOKEN", None)

        with pytest.raises(ValueError):
            GradientTextEmbedder(workspace_id=workspace_id)

    @pytest.mark.unit
    def test_init_without_workspace(self):
        access_token = "access_token"
        os.environ.pop("GRADIENT_WORKSPACE_ID", None)

        with pytest.raises(ValueError):
            GradientTextEmbedder(access_token=access_token)

    @pytest.mark.unit
    def test_init_from_params(self):
        access_token = "access_token"
        workspace_id = "workspace_id"

        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_from_params_precedence(self):
        access_token = "access_token"
        workspace_id = "workspace_id"

        os.environ["GRADIENT_ACCESS_TOKEN"] = "env_access_token"
        os.environ["GRADIENT_WORKSPACE_ID"] = "env_workspace_id"

        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token
