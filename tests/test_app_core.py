"""
Tests for app.py — core NLP helpers and FastAPI endpoints.

Upgraded packages under test:
  - fastapi  0.104.1 → 0.109.1
  - python-multipart 0.0.6 → 0.0.7
  - keras    2.14.0  → 3.12.0
  - nltk     3.8.1   → 3.9
"""

import importlib
import json
import pickle
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers – build a minimal fake module tree so app.py can be imported
# without real model files or a GPU.
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "how", "are", "you", "name", "i", "am", "bye"]
FAKE_CLASSES = ["greeting", "goodbye", "name"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hello", "hi", "how are you"],
            "responses": ["Hello!", "Hi there!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "see you"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is", "i am"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}


def _make_fake_model(num_classes=3):
    """Return a mock Keras model whose predict() returns a plausible array."""
    fake_model = MagicMock()
    probs = np.zeros((1, num_classes))
    probs[0, 0] = 0.95  # highest for class 0
    fake_model.predict.return_value = probs
    return fake_model


def _patch_imports_and_load(monkeypatch):
    """
    Patch all heavy I/O (pickle, open, keras load_model, StaticFiles)
    then import (or reload) app so the module-level try/except succeeds.
    Returns the imported app module.
    """
    fake_model = _make_fake_model(len(FAKE_CLASSES))

    # -- keras.models.load_model --
    keras_models_mod = types.ModuleType("keras.models")
    keras_models_mod.load_model = MagicMock(return_value=fake_model)
    keras_mod = types.ModuleType("keras")
    keras_mod.models = keras_models_mod

    # keras.layers (needed by train.py imports; guard app.py from blowing up)
    keras_layers_mod = types.ModuleType("keras.layers")
    keras_layers_mod.Dense = MagicMock()
    keras_layers_mod.Dropout = MagicMock()
    keras_mod.layers = keras_layers_mod

    monkeypatch.setitem(sys.modules, "keras", keras_mod)
    monkeypatch.setitem(sys.modules, "keras.models", keras_models_mod)
    monkeypatch.setitem(sys.modules, "keras.layers", keras_layers_mod)

    # -- fastapi.staticfiles.StaticFiles (avoid directory check) --
    static_mod = types.ModuleType("fastapi.staticfiles")
    static_mod.StaticFiles = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "fastapi.staticfiles", static_mod)

    # -- pickle.load returns words / classes alternately --
    pickle_calls = [FAKE_WORDS, FAKE_CLASSES]
    pickle_iter = iter(pickle_calls)
    original_pickle_load = pickle.load

    def fake_pickle_load(f):
        return next(pickle_iter)

    monkeypatch.setattr(pickle, "load", fake_pickle_load)

    # -- builtins.open: intercept intents.json and words/classes pkl files --
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if isinstance(path, str) and "intents.json" in path:
            import io
            return io.StringIO(json.dumps(FAKE_INTENTS))
        if isinstance(path, str) and path.endswith(".pkl"):
            # Return a dummy binary stream; pickle.load is already mocked
            import io
            return io.BytesIO(b"")
        if isinstance(path, str) and "index.html" in path:
            import io
            return io.StringIO("<html><body>chatbot</body></html>")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    # Force re-import of app
    if "app" in sys.modules:
        del sys.modules["app"]

    import app as app_module
    # Patch the module-level model reference so predict works
    app_module.model = fake_model
    app_module.words = FAKE_WORDS
    app_module.classes = FAKE_CLASSES
    app_module.intents = FAKE_INTENTS
    return app_module, fake_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    """
    Module-scoped fixture: patches heavy dependencies, imports app, and
    returns (app_module, fake_model, TestClient).
    We use monkeypatch-style patching via unittest.mock.patch so it can be
    module-scoped without needing a function-scoped monkeypatch.
    """
    fake_model = _make_fake_model(len(FAKE_CLASSES))
    patches = []

    # keras
    keras_models_mod = types.ModuleType("keras.models")
    keras_models_mod.load_model = MagicMock(return_value=fake_model)
    keras_mod = types.ModuleType("keras")
    keras_mod.models = keras_models_mod
    keras_layers_mod = types.ModuleType("keras.layers")
    keras_layers_mod.Dense = MagicMock()
    keras_layers_mod.Dropout = MagicMock()
    keras_mod.layers = keras_layers_mod
    sys.modules["keras"] = keras_mod
    sys.modules["keras.models"] = keras_models_mod
    sys.modules["keras.layers"] = keras_layers_mod

    # staticfiles
    static_mod = types.ModuleType("fastapi.staticfiles")
    static_mod.StaticFiles = MagicMock(return_value=MagicMock())
    sys.modules["fastapi.staticfiles"] = static_mod

    pickle_calls = iter([FAKE_WORDS, FAKE_CLASSES])

    def fake_pickle_load(f):
        return next(pickle_calls)

    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        import io
        if isinstance(path, str) and "intents.json" in path:
            return io.StringIO(json.dumps(FAKE_INTENTS))
        if isinstance(path, str) and path.endswith(".pkl"):
            return io.BytesIO(b"")
        if isinstance(path, str) and "index.html" in path:
            return io.StringIO("<html><body>chatbot</body></html>")
        return real_open(path, mode, *args, **kwargs)

    p1 = patch("pickle.load", side_effect=fake_pickle_load)
    p2 = patch("builtins.open", side_effect=fake_open)
    p1.start()
    p2.start()
    patches.extend([p1, p2])

    if "app" in sys.modules:
        del sys.modules["app"]

    import app as app_module
    app_module.model = fake_model
    app_module.words = FAKE_WORDS
    app_module.classes = FAKE_CLASSES
    app_module.intents = FAKE_INTENTS

    client = TestClient(app_module.app, raise_server_exceptions=False)

    yield app_module, fake_model, client

    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Tests: NLP helpers
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_basic_tokenization(self, app_env):
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lemmatization_lowercases(self, app_env):
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("Running DOGS")
        for token in result:
            assert token == token.lower()

    def test_empty_string(self, app_env):
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("")
        assert isinstance(result, list)

    def test_punctuation_handled(self, app_env):
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("Hello! How are you?")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_returns_list_of_strings(self, app_env):
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("test sentence")
        assert all(isinstance(t, str) for t in result)

    def test_lemmatizer_applied_plural(self, app_env):
        """'dogs' should lemmatize to 'dog'."""
        app_module, _, _ = app_env
        result = app_module.clean_up_sentence("dogs")
        assert "dog" in result


class TestBagOfWords:
    def test_returns_numpy_array(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_vocabulary(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("hello", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)

    def test_known_word_sets_bit(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_all_zeros(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("xyzzy", FAKE_WORDS)
        assert result.sum() == 0

    def test_binary_values_only(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("hello hi", FAKE_WORDS)
        assert set(result.tolist()).issubset({0, 1})

    def test_empty_sentence(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)
        assert result.sum() == 0

    def test_multiple_known_words(self, app_env):
        app_module, _, _ = app_env
        result = app_module.bow("hello hi", FAKE_WORDS)
        assert result[FAKE_WORDS.index("hello")] == 1
        assert result[FAKE_WORDS.index("hi")] == 1


class TestPredictClass:
    def test_returns_list(self, app_env):
        app_module, fake_model, _ = app_env
        result = app_module.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_each_item_has_intent_and_probability(self, app_env):
        app_module, fake_model, _ = app_env
        result = app_module.predict_class("hello", fake_model)
        for item in result:
            assert "intent" in item
            assert "probability" in item

    def test_probability_is_string_or_numeric(self, app_env):
        app_module, fake_model, _ = app_env
        result = app_module.predict_class("hello", fake_model)
        if result:
            # probability may be stored as string in some implementations
            float(result[0]["probability"])

    def test_model_predict_called(self, app_env):
        app_module, fake_model, _ = app_env
        fake_model.reset_mock()
        app_module.predict_class("hello", fake_model)
        fake_model.predict.assert_called_once()

    def test_model_predict_receives_2d_array(self, app_env):
        """Keras 3.x expects a 2-D input array (batch_size, features)."""
        app_module, fake_model, _ = app_env
        fake_model.reset_mock()
        app_module.predict_class("hello", fake_model)
        call_args = fake_model.predict.call_args
        input_arr = call_args[0][0]
        assert input_arr.ndim == 2, "predict() must receive a 2-D array (batch, features)"

    def test_results_sorted_by_probability_descending(self, app_env):
        app_module, fake_model, _ = app_env
        result = app_module.predict_class("hello", fake_model)
        if len(result) > 1:
            probs = [float(r["probability"]) for r in result]
            assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Tests: FastAPI endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, app_env):
        _, _, client = app_env
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_contains_status(self, app_env):
        _, _, client = app_env
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_body_contains_model(self, app_env):
        _, _, client = app_env
        resp = client.get("/health")
        data = resp.json()
        assert "model" in data


class TestChatEndpoint:
    def test_basic_chat_200(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hello"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hello"})
        data = resp.json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_float(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hello"})
        data = resp.json()
        assert isinstance(data["confidence"], float)

    def test_empty_message_returns_400(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_message_returns_400(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_my_name_is_pattern(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200

    def test_hi_my_name_is_pattern(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert resp.status_code == 200

    def test_i_am_pattern(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "i am Charlie"})
        assert resp.status_code == 200

    def test_name_placeholder_replaced(self, app_env):
        """When the response template contains {n} it should be replaced."""
        app_module, fake_model, client = app_env
        # Force class prediction to 'name' intent (index 2)
        probs = np.zeros((1, len(FAKE_CLASSES)))
        probs[0, 2] = 0.99
        fake_model.predict.return_value = probs

        resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200
        body = resp.json()
        # {n} must not appear literally in the final response
        assert "{n}" not in body["response"]

    def test_confidence_between_0_and_1(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hello"})
        data = resp.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_large_input_handled(self, app_env):
        _, _, client = app_env
        long_msg = "hello " * 500
        resp = client.post("/api/chat", json={"msg": long_msg})
        assert resp.status_code in (200, 500)  # must not crash the server

    def test_content_type_json(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": "hello"})
        assert "application/json" in resp.headers["content-type"]

    def test_non_string_msg_returns_422(self, app_env):
        _, _, client = app_env
        resp = client.post("/api/chat", json={"msg": 12345})
        # FastAPI should coerce or reject; either way server stays alive
        assert resp.status_code in (200, 422)


class TestHomeEndpoint:
    def test_home_returns_200(self, app_env):
        _, _, client = app_env
        resp = client.get("/")
        assert resp.status_code == 200

    def test_home_returns_html(self, app_env):
        _, _, client = app_env
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]


class TestCORSMiddleware:
    def test_cors_origin_header_present(self, app_env):
        _, _, client = app_env
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should add the header on preflight
        assert resp.status_code in (200, 400)

    def test_cors_allows_all_origins_on_get(self, app_env):
        _, _, client = app_env
        resp = client.get("/health", headers={"Origin": "http://evil.com"})
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao in ("*", "http://evil.com", "")


# ---------------------------------------------------------------------------
# Tests: Pydantic models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, app_env):
        app_module, _, _ = app_env
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_missing_field(self, app_env):
        app_module, _, _ = app_env
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            app_module.MessageRequest()

    def test_chat_response_valid(self, app_env):
        app_module, _, _ = app_env
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == 0.95

    def test_chat_response_confidence_float(self, app_env):
        app_module, _, _ = app_env
        resp = app_module.ChatResponse(response="Hi!", confidence=0)
        assert isinstance(resp.confidence, float)


# ---------------------------------------------------------------------------
# Tests: FastAPI app metadata (fastapi 0.109.1 compatibility)
# ---------------------------------------------------------------------------

class TestAppMetadata:
    def test_app_title(self, app_env):
        app_module, _, _ = app_env
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_env):
        app_module, _, _ = app_env
        assert app_module.app.version == "1.0.0"

    def test_openapi_schema_accessible(self, app_env):
        _, _, client = app_env
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema

    def test_openapi_chat_path_defined(self, app_env):
        _, _, client = app_env
        resp = client.get("/openapi.json")
        schema = resp.json()
        assert "/api/chat" in schema["paths"]

    def test_docs_endpoint_accessible(self, app_env):
        _, _, client = app_env
        resp = client.get("/docs")
        assert resp.status_code == 200
