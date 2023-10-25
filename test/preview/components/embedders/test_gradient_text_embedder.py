import os
import pytest
from haystack.preview.components.embedders.gradient_text_embedder import GradientTextEmbedder

access_token = "access_token"
workspace_id = "workspace_id"


class TestGradientTextEmbedder:
    @pytest.mark.unit
    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", access_token)
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", workspace_id)

        embedder = GradientTextEmbedder()
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_without_access_token(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_ACCESS_TOKEN", raising=True)

        with pytest.raises(ValueError):
            GradientTextEmbedder(workspace_id=workspace_id)

    @pytest.mark.unit
    def test_init_without_workspace(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_WORKSPACE_ID", raising=True)

        with pytest.raises(ValueError):
            GradientTextEmbedder(access_token=access_token)

    @pytest.mark.unit
    def test_init_from_params(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_from_params_precedence(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", "env_access_token")
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", "env_workspace_id")

        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_to_dict(self):
        component = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        data = component.to_dict()
        assert data == {
            "type": "GradientTextEmbedder",
            "init_parameters": {"workspace_id": workspace_id, "model_name": "bge-large"},
        }
