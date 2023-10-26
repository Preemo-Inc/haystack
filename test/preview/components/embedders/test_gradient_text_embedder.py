import pytest
from haystack.preview.components.embedders.gradient_text_embedder import GradientTextEmbedder
from unittest.mock import MagicMock, patch
import numpy as np


access_token = "access_token"
workspace_id = "workspace_id"


def has_gradient():
    try:
        import gradientai

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not has_gradient(), reason="Gradient is not installed")
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

    @pytest.mark.unit
    def test_warmup(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with("bge-large")

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock(default_return_value="fake model")
        embedder.warm_up()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with("bge-large")

    # @pytest.mark.unit
    # def test_run(self):
    #     model = "bge-large"
    #     embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
    #     embedder._gradient.create_embedding = MagicMock(return_value=np.zeros(1024))
    #     embedder._gradient.create_embedding.assert_called_once_with(model=model)

    #     result = embedder.run(text="The food was delicious")

    #     assert len(result["embedding"]) == 1024  # 1024 is the bge-large embedding size
    #     assert all(isinstance(x, float) for x in result["embedding"])
    #     assert result["metadata"] == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}
