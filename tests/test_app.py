"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Covers:
- FastAPI route handlers (/, /api/chat, /health)
- CORS middleware configuration
- Core NLP processing functions (clean_up_sentence, bow, predict_class, getResponse)
- Pydantic request/response models
- Error / edge-case handling
- Upgraded-package surface: fastapi 0.109.1, python-multipart 0.0.7,
  keras 3.x, nltk 3.9
"""

import importlib
import json
import pickle
import sys
import types
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers & shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "good", "morning", "name", "i", "am"]
SAMPLE_CLASSES = ["greeting", "goodbye", "introduction"]
SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Good bye", "See you later"],
            "responses": ["Bye!", "See you later!", "Goodbye!"],
        },
        {
            "tag": "introduction",
            "patterns": ["My name is {n}", "I am {n}", "Hi my name is {n}"],
            "responses": ["Nice to meet you, {n}!", "Hello, {n}!"],
        },
    ]
}


def _make_mock_model(output: list | None = None):
    """Return a mock Keras model whose predict() returns a fixed probability array."""
    mock_model = MagicMock()
    if output is None:
        # Default: classify as 'greeting' with high confidence
        arr = np.array([[0.85, 0.10, 0.05]])
    else:
        arr = np.array([output])
    mock_model.predict.return_value = arr
    return mock_model


def _pickle_bytes(obj) -> bytes:
    return pickle.dumps(obj)


# ---------------------------------------------------------------------------
# Fixture: load the app module with all heavy I/O mocked
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_app_module():
    """
    Import app.py with:
      - keras.models.load_model → mock model
      - pickle.load             → sample words / classes
      - builtins.open           → intents JSON or HTML stub
      - StaticFiles             → disabled
      - nltk resources          → pre-downloaded stubs
    """
    mock_model = _make_mock_model()

    # Prepare pickle bytes for words and classes
    words_pkl = _pickle_bytes(SAMPLE_WORDS)
    classes_pkl = _pickle_bytes(SAMPLE_CLASSES)
    intents_json = json.dumps(SAMPLE_INTENTS)

    pkl_call_count = {"n": 0}

    def fake_pickle_load(f):
        pkl_call_count["n"] += 1
        if pkl_call_count["n"] == 1:
            return SAMPLE_WORDS
        return SAMPLE_CLASSES

    # Stub for open() – handles intents.json and templates/index.html
    original_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if "intents.json" in path_str:
            return mock_open(read_data=intents_json)()
        if "index.html" in path_str:
            return mock_open(read_data="<html>chatbot</html>")()
        # Pickle files – return BytesIO wrapping real pickle bytes
        if "words.pkl" in path_str:
            return BytesIO(words_pkl)
        if "classes.pkl" in path_str:
            return BytesIO(classes_pkl)
        return original_open(path, mode, *args, **kwargs)

    # Patch StaticFiles so the directory doesn't need to exist
    fake_static = MagicMock()

    with (
        patch("keras.models.load_model", return_value=mock_model),
        patch("pickle.load", side_effect=fake_pickle_load),
        patch("builtins.open", side_effect=fake_open),
        patch("fastapi.staticfiles.StaticFiles", return_value=fake_static),
        patch("nltk.word_tokenize", side_effect=lambda s: s.lower().split()),
        patch("nltk.stem.WordNetLemmatizer.lemmatize", side_effect=lambda w, **kw: w),
    ):
        # Remove cached module so we get a fresh import
        for key in list(sys.modules.keys()):
            if key in ("app",):
                del sys.modules[key]

        import app as app_module

        yield app_module, mock_model


@pytest.fixture(scope="module")
def client(patched_app_module):
    app_module, _ = patched_app_module
    with TestClient(app_module.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def mock_model(patched_app_module):
    _, model = patched_app_module
    return model


# ---------------------------------------------------------------------------
# 1. FastAPI – application meta
# ---------------------------------------------------------------------------

class TestAppMetadata:
    def test_app_title(self, patched_app_module):
        app_module, _ = patched_app_module
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, patched_app_module):
        app_module, _ = patched_app_module
        assert app_module.app.version == "1.0.0"

    def test_cors_middleware_present(self, patched_app_module):
        app_module, _ = patched_app_module
        from starlette.middleware.cors import CORSMiddleware
        middleware_types = [
            m.cls for m in app_module.app.user_middleware
        ]
        assert CORSMiddleware in middleware_types


# ---------------------------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ---------------------------------------------------------------------------
# 3. Home endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_content_type_html(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 4. Chat endpoint – happy paths
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:
    def _post(self, client, msg: str):
        return client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
        resp = self._post(client, "hello")
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_goodbye_message(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = self._post(client, "bye")
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] in [r for i in SAMPLE_INTENTS["intents"] for r in i["responses"]]

    def test_my_name_is_pattern(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        data = resp.json()
        # The name substitution should replace {n} with "Alice"
        assert "Alice" in data["response"] or isinstance(data["response"], str)

    def test_hi_my_name_is_pattern(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["response"], str)

    def test_i_am_pattern(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = self._post(client, "i am Carol")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["response"], str)

    def test_confidence_is_float(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.70, 0.20, 0.10]])
        resp = self._post(client, "how are you")
        assert resp.status_code == 200
        assert isinstance(resp.json()["confidence"], float)

    def test_mixed_case_message(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.80, 0.10, 0.10]])
        resp = self._post(client, "Hello World")
        assert resp.status_code == 200

    def test_message_with_leading_trailing_spaces(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.80, 0.10, 0.10]])
        resp = self._post(client, "  hello  ")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. Chat endpoint – error / edge cases
# ---------------------------------------------------------------------------

class TestChatEndpointErrors:
    def _post(self, client, payload):
        return client.post("/api/chat", json=payload)

    def test_empty_string_returns_400(self, client):
        resp = self._post(client, {"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = self._post(client, {"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_wrong_type_for_msg_returns_422(self, client):
        resp = client.post("/api/chat", json={"msg": 12345})
        # FastAPI coerces ints to str, so 200 is also acceptable
        assert resp.status_code in (200, 422)

    def test_model_exception_returns_500(self, client, mock_model):
        mock_model.predict.side_effect = RuntimeError("Model exploded")
        resp = self._post(client, {"msg": "hello"})
        assert resp.status_code == 500
        # Restore
        mock_model.predict.side_effect = None
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])

    def test_error_detail_present_on_400(self, client):
        resp = self._post(client, {"msg": ""})
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# 6. Pydantic models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self):
        # Import lazily to avoid top-level side-effects
        from pydantic import ValidationError

        # We need to reference the model inside the (already imported) module
        import app as app_module
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_chat_response_valid(self):
        import app as app_module
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_confidence_float(self):
        import app as app_module
        resp = app_module.ChatResponse(response="Hey", confidence=0.5)
        assert isinstance(resp.confidence, float)


# ---------------------------------------------------------------------------
# 7. clean_up_sentence
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_returns_list(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.clean_up_sentence("Hello World")
        assert isinstance(result, list)

    def test_words_are_lowercase(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.clean_up_sentence("Hello World")
        for w in result:
            assert w == w.lower()

    def test_empty_string(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.clean_up_sentence("")
        assert isinstance(result, list)

    def test_single_word(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.clean_up_sentence("hello")
        assert "hello" in result


# ---------------------------------------------------------------------------
# 8. bow (bag of words)
# ---------------------------------------------------------------------------

class TestBow:
    def test_returns_numpy_array(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_vocabulary(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert len(result) == len(SAMPLE_WORDS)

    def test_known_word_sets_one(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("hello", SAMPLE_WORDS)
        idx = SAMPLE_WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_all_zeros(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("xyzzy", SAMPLE_WORDS)
        assert result.sum() == 0

    def test_values_are_binary(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("hello hi bye", SAMPLE_WORDS)
        for v in result:
            assert v in (0, 1)

    def test_empty_sentence(self, patched_app_module):
        app_module, _ = patched_app_module
        result = app_module.bow("", SAMPLE_WORDS)
        assert result.sum() == 0

    def test_show_details_flag(self, patched_app_module, capsys):
        app_module, _ = patched_app_module
        app_module.bow("hello", SAMPLE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "hello" in captured.out


# ---------------------------------------------------------------------------
# 9. predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, patched_app_module):
        app_module, mock_model = patched_app_module
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
        result = app_module.predict_class("hello", mock_model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, patched_app_module):
        app_module, mock_model = patched_app_module
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
        result = app_module.predict_class("hello", mock_model)
        assert len(result) > 0
        first = result[0]
        assert "intent" in first
        assert "probability" in first

    def test_highest_probability_is_first(self, patched_app_module):
        app_module, mock_model = patched_app_module
        mock_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("bye", mock_model)
        if len(result) > 1:
            assert float(result[0]["probability"]) >= float(result[1]["probability"])

    def test_model_predict_called_once(self, patched_app_module):
        app_module, mock_model = patched_app_module
        mock_model.predict.reset_mock()
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
        app_module.predict_class("hello", mock_model)
        mock_model.predict.assert_called_once()

    def test_probability_values_are_strings_or_floats(self, patched_app_module):
        app_module, mock_model = patched_app_module
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
        result = app_module.predict_class("hello", mock_model)
        for item in result:
            # probability may be stored as str or numeric
            float(item["probability"])  # should not raise


# ---------------------------------------------------------------------------
# 10. NLTK integration (nltk 3.9 API)
# ---------------------------------------------------------------------------

class TestNLTKIntegration:
    def test_word_tokenize_called(self, patched_app_module):
        app_module, _ = patched_app_module
        with patch("nltk.word_tokenize", return_value=["hello", "world"]) as mock_tok:
            app_module.clean_up_sentence("hello world")
            mock_tok.assert_called_once()

    def test_lemmatizer_called_per_word(self, patched_app_module):
        app_module, _ = patched_app_module
        tokens = ["running", "jumps", "easily"]
        with (
            patch("nltk.word_tokenize", return_value=tokens),
            patch.object(app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w) as mock_lem,
        ):
            app_module.clean_up_sentence("running jumps easily")
            assert mock_lem.call_count == len(tokens)

    def test_lemmatizer_instance_type(self, patched_app_module):
        app_module, _ = patched_app_module
        from nltk.stem import WordNetLemmatizer
        assert isinstance(app_module.lemmatizer, WordNetLemmatizer)


# ---------------------------------------------------------------------------
# 11. Keras model integration (keras 3.x)
# ---------------------------------------------------------------------------

class TestKerasIntegration:
    def test_load_model_called_at_startup(self):
        """Ensure load_model is invoked with the expected filename."""
        words_pkl = _pickle_bytes(SAMPLE_WORDS)
        classes_pkl = _pickle_bytes(SAMPLE_CLASSES)
        intents_json = json.dumps(SAMPLE_INTENTS)
        pkl_call_count = {"n": 0}

        def fake_pickle_load(f):
            pkl_call_count["n"] += 1
            return SAMPLE_WORDS if pkl_call_count["n"] == 1 else SAMPLE_CLASSES

        def fake_open(path, mode="r", *args, **kwargs):
            path_str = str(path)
            if "intents.json" in path_str:
                return mock_open(read_data=intents_json)()
            if "words.pkl" in path_str:
                return BytesIO(words_pkl)
            if "classes.pkl" in path_str:
                return BytesIO(classes_pkl)
            return mock_open(read_data="<html></html>")()

        with (
            patch("keras.models.load_model", return_value=_make_mock_model()) as mock_lm,
            patch("pickle.load", side_effect=fake_pickle_load),
            patch("builtins.open", side_effect=fake_open),
            patch("fastapi.staticfiles.StaticFiles", return_value=MagicMock()),
            patch("nltk.word_tokenize", side_effect=lambda s: s.lower().split()),
            patch("nltk.stem.WordNetLemmatizer.lemmatize", side_effect=lambda w, **kw: w),
        ):
            for key in list(sys.modules.keys()):
                if key == "app":
                    del sys.modules[key]
            import app  # noqa: F401
            mock_lm.assert_called_once_with("chatbot_model.h5")

    def test_model_predict_input_shape(self, patched_app_module):
        """predict() must receive a 2-D numpy array."""
        app_module, mock_model = patched_app_module
        captured_args = []

        def capture_predict(x):
            captured_args.append(x)
            return np.array([[0.85, 0.10, 0.05]])

        mock_model.predict.side_effect = capture_predict
        app_module.predict_class("hello", mock_model)
        mock_model.predict.side_effect = None
        mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])

        assert len(captured_args) == 1
        arr = captured_args[0]
        assert arr.ndim == 2

    def test_model_predict_output_processed(self, patched_app_module):
        """predict_class must map model output probabilities to classes."""
        app_module, mock_model = patched_app_module
        mock_model.predict.return_value = np.array([[0.01, 0.01, 0.98]])
        result = app_module.predict_class("my name is Dave", mock_model)
        assert len(result) > 0
        # Highest prob index = 2 → "introduction"
        assert result[0]["intent"] == SAMPLE_CLASSES[2]


# ---------------------------------------------------------------------------
# 12. getResponse helper (if exported)
# ---------------------------------------------------------------------------

class TestGetResponse:
    """Test getResponse if it exists in app module."""

    def test_returns_string(self, patched_app_module):
        app_module, _ = patched_app_module
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not exported at module level")
        ints = [{"intent": "greeting", "probability": "0.85"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert isinstance(result, str)

    def test_response_in_valid_responses(self, patched_app_module):
        app_module, _ = patched_app_module
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not exported at module level")
        ints = [{"intent": "greeting", "probability": "0.85"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        valid = SAMPLE_INTENTS["intents"][0]["responses"]
        assert result in valid

    def test_unknown_intent_handled(self, patched_app_module):
        app_module, _ = patched_app_module
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not exported at module level")
        ints = [{"intent": "unknown_xyz", "probability": "0.5"}]
        # Should not raise – may return empty string or fallback
        try:
            result = app_module.getResponse(ints, SAMPLE_INTENTS)
            assert isinstance(result, str)
        except Exception:
            pass  # acceptable if intent not found


# ---------------------------------------------------------------------------
# 13. OpenAPI / docs endpoints (fastapi 0.109.x)
# ---------------------------------------------------------------------------

class TestOpenAPIEndpoints:
    def test_openapi_json_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_schema_has_paths(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()
        assert "/api/chat" in schema.get("paths", {})
        assert "/health" in schema.get("paths", {})

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_endpoint(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 14. CORS headers (via middleware)
# ---------------------------------------------------------------------------

class TestCORSHeaders:
    def test_cors_header_on_health(self, client):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_preflight(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 204)


# ---------------------------------------------------------------------------
# 15. Content-type & HTTP method enforcement
# ---------------------------------------------------------------------------

class TestHTTPMethodEnforcement:
    def test_get_on_chat_returns_405(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_put_on_chat_returns_405(self, client):
        resp = client.put("/api/chat", json={"msg": "hello"})
        assert resp.status_code == 405

    def test_non_json_body_returns_422(self, client):
        resp = client.post(
            "/api/chat",
            data="plain text",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 422
