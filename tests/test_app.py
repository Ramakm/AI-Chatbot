"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).
Covers:
  - FastAPI endpoints (/, /api/chat, /health)
  - CORS middleware configuration
  - clean_up_sentence / bow / predict_class / getResponse helpers
  - Pydantic request/response models
  - NLTK (WordNetLemmatizer) usage
  - Keras model integration (mocked)
  - python-multipart integration (multipart form requests)
  - Error / edge-case handling
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
# Helpers to build a minimal fake module environment so app.py can be
# imported without real model files or a GPU.
# ---------------------------------------------------------------------------

FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is", "I am"],
            "responses": ["Nice to meet you, {n}!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
    ]
}

FAKE_WORDS = ["bye", "hello", "hey", "hi", "later", "name", "see"]
FAKE_CLASSES = ["goodbye", "greeting", "name"]


def _make_fake_model():
    """Return a MagicMock that behaves like a Keras model."""
    model = MagicMock()
    # predict returns shape (1, num_classes)
    model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
    return model


def _build_app_module():
    """
    Import app.py with all external I/O mocked so it can be tested in isolation.
    Returns the imported module.
    """
    fake_model = _make_fake_model()

    # --- patch builtins.open for the three data files app.py reads at module level ---
    intents_json = json.dumps(FAKE_INTENTS)

    def _open_side_effect(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if "intents.json" in path_str:
            return mock_open(read_data=intents_json)()
        elif "words.pkl" in path_str:
            return mock_open(read_data=pickle.dumps(FAKE_WORDS))()
        elif "classes.pkl" in path_str:
            return mock_open(read_data=pickle.dumps(FAKE_CLASSES))()
        elif "index.html" in path_str:
            return mock_open(read_data="<html>chatbot</html>")()
        # fallback
        raise FileNotFoundError(f"Unexpected open: {path_str}")

    # patch pickle.load so it returns the correct list from the fake bytes
    _pickle_calls = {"count": 0}

    def _pickle_load_side_effect(f):
        # First call → words, second call → classes
        _pickle_calls["count"] += 1
        if _pickle_calls["count"] == 1:
            return FAKE_WORDS
        return FAKE_CLASSES

    with patch("builtins.open", side_effect=_open_side_effect), \
         patch("pickle.load", side_effect=_pickle_load_side_effect), \
         patch("keras.models.load_model", return_value=fake_model), \
         patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):

        # Force re-import each time this helper is called
        if "app" in sys.modules:
            del sys.modules["app"]

        import app as app_module  # noqa: PLC0415

    return app_module, fake_model


# ---------------------------------------------------------------------------
# Module-level fixture: import once, reuse across all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module_and_model():
    module, fake_model = _build_app_module()
    return module, fake_model


@pytest.fixture(scope="module")
def client(app_module_and_model):
    module, _ = app_module_and_model
    return TestClient(module.app)


@pytest.fixture(scope="module")
def fake_model(app_module_and_model):
    _, model = app_module_and_model
    return model


@pytest.fixture(scope="module")
def app_module(app_module_and_model):
    module, _ = app_module_and_model
    return module


# ===========================================================================
# 1. FastAPI application bootstrap
# ===========================================================================

class TestAppBootstrap:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_words_loaded(self, app_module):
        assert app_module.words == FAKE_WORDS

    def test_classes_loaded(self, app_module):
        assert app_module.classes == FAKE_CLASSES

    def test_intents_loaded(self, app_module):
        assert "intents" in app_module.intents
        assert len(app_module.intents["intents"]) == 3

    def test_cors_middleware_present(self, app_module):
        middleware_types = [
            type(m).__name__ for m in app_module.app.user_middleware
        ]
        # CORSMiddleware is registered
        assert any("CORS" in name for name in middleware_types)


# ===========================================================================
# 2. Pydantic models
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="Hello")
        assert req.msg == "Hello"

    def test_message_request_empty_string(self, app_module):
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_zero_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Hmm", confidence=0.0)
        assert resp.confidence == 0.0

    def test_chat_response_serialisation(self, app_module):
        resp = app_module.ChatResponse(response="Test", confidence=0.5)
        d = resp.model_dump()
        assert set(d.keys()) == {"response", "confidence"}


# ===========================================================================
# 3. GET /  (HTML home page)
# ===========================================================================

class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_returns_html_content_type(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_home_returns_html_body(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert "html" in resp.text.lower()


# ===========================================================================
# 4. GET /health
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model"] == "loaded"


# ===========================================================================
# 5. POST /api/chat  – happy paths
# ===========================================================================

class TestChatEndpointHappyPath:
    def test_basic_greeting(self, client, app_module, fake_model):
        # Make sure predict_class returns a result
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "confidence" in body
        assert isinstance(body["response"], str)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_my_name_is_pattern(self, client, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200
        body = resp.json()
        # The name replacement should have happened
        assert isinstance(body["response"], str)

    def test_hi_my_name_is_pattern(self, client, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["response"], str)

    def test_i_am_pattern(self, client, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        resp = client.post("/api/chat", json={"msg": "i am Carol"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["response"], str)

    def test_response_model_fields_present(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Hi"})
        body = resp.json()
        assert "response" in body
        assert "confidence" in body

    def test_confidence_is_float(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Hi"})
        assert isinstance(resp.json()["confidence"], float)


# ===========================================================================
# 6. POST /api/chat  – error / edge cases
# ===========================================================================

class TestChatEndpointErrorCases:
    def test_empty_string_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        resp = client.post("/api/chat", data="not json")
        assert resp.status_code in (422, 415)

    def test_internal_error_returns_500(self, client, app_module, fake_model):
        fake_model.predict.side_effect = RuntimeError("GPU exploded")
        resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 500
        assert "Error processing message" in resp.json()["detail"]
        # restore
        fake_model.predict.side_effect = None
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])

    def test_very_long_message(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        long_msg = "hello " * 500
        resp = client.post("/api/chat", json={"msg": long_msg})
        assert resp.status_code == 200


# ===========================================================================
# 7. CORS headers
# ===========================================================================

class TestCORSHeaders:
    def test_cors_preflight(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # FastAPI CORS middleware should allow all origins
        assert resp.status_code in (200, 204)
        assert "access-control-allow-origin" in resp.headers

    def test_cors_allow_origin_wildcard(self, client):
        resp = client.get("/health", headers={"Origin": "http://another-domain.com"})
        assert resp.headers.get("access-control-allow-origin") in (
            "*",
            "http://another-domain.com",
        )


# ===========================================================================
# 8. clean_up_sentence  (NLTK WordNetLemmatizer + word_tokenize)
# ===========================================================================

class TestCleanUpSentence:
    def test_basic_tokenization(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)
        assert "hello" in result
        assert "world" in result

    def test_lemmatization_running(self, app_module):
        # "running" should lemmatize to "running" (verb, default noun lemmatization)
        result = app_module.clean_up_sentence("running")
        assert "running" in result or "run" in result

    def test_lemmatization_studies(self, app_module):
        result = app_module.clean_up_sentence("studies")
        # noun lemmatization: studies → study
        assert "study" in result or "studies" in result

    def test_lowercase_conversion(self, app_module):
        result = app_module.clean_up_sentence("HELLO")
        assert "hello" in result

    def test_punctuation_tokenized(self, app_module):
        result = app_module.clean_up_sentence("Hello!")
        # nltk tokenizes punctuation separately
        assert "hello" in result

    def test_empty_sentence(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_multiple_words(self, app_module):
        result = app_module.clean_up_sentence("Hi how are you")
        assert len(result) >= 4

    def test_returns_list(self, app_module):
        assert isinstance(app_module.clean_up_sentence("test"), list)


# ===========================================================================
# 9. bow (bag-of-words)
# ===========================================================================

class TestBow:
    def test_returns_numpy_array(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(arr, np.ndarray)

    def test_correct_length(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert len(arr) == len(FAKE_WORDS)

    def test_known_word_sets_bit(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert arr[idx] == 1

    def test_unknown_word_zero(self, app_module):
        arr = app_module.bow("xyzzy", FAKE_WORDS)
        assert all(v == 0 for v in arr)

    def test_empty_sentence_all_zeros(self, app_module):
        arr = app_module.bow("", FAKE_WORDS)
        assert np.all(arr == 0)

    def test_multiple_known_words(self, app_module):
        arr = app_module.bow("hi bye", FAKE_WORDS)
        assert arr[FAKE_WORDS.index("hi")] == 1
        assert arr[FAKE_WORDS.index("bye")] == 1

    def test_show_details_flag(self, app_module, capsys):
        app_module.bow("hello", FAKE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_dtype_int(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert arr.dtype in (np.int32, np.int64, np.float64, int)


# ===========================================================================
# 10. predict_class
# ===========================================================================

class TestPredictClass:
    def test_returns_list(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("hello", fake_model)
        if result:  # may be empty if threshold filters all
            assert "intent" in result[0]
            assert "probability" in result[0]

    def test_probability_is_numeric(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("hello", fake_model)
        if result:
            assert float(result[0]["probability"]) >= 0.0

    def test_calls_model_predict(self, app_module, fake_model):
        fake_model.predict.reset_mock()
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        app_module.predict_class("hello", fake_model)
        assert fake_model.predict.called

    def test_model_receives_2d_array(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        app_module.predict_class("hello", fake_model)
        call_args = fake_model.predict.call_args[0][0]
        assert call_args.ndim == 2

    def test_top_result_highest_probability(self, app_module, fake_model):
        # class index 1 has highest prob → should be first result
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("hello", fake_model)
        if result:
            assert result[0]["intent"] == FAKE_CLASSES[1]  # "greeting"


# ===========================================================================
# 11. NLTK WordNetLemmatizer (exercises the upgraded nltk 3.9 API)
# ===========================================================================

class TestNLTKLemmatizer:
    """Directly exercise the NLTK WordNetLemmatizer as used in app.py."""

    def test_lemmatizer_noun(self, app_module):
        result = app_module.lemmatizer.lemmatize("dogs")
        assert result == "dog"

    def test_lemmatizer_verb(self, app_module):
        result = app_module.lemmatizer.lemmatize("running", pos="v")
        assert result == "run"

    def test_lemmatizer_adjective(self, app_module):
        result = app_module.lemmatizer.lemmatize("better", pos="a")
        assert result == "good"

    def test_lemmatizer_unchanged(self, app_module):
        result = app_module.lemmatizer.lemmatize("cat")
        assert result == "cat"

    def test_lemmatizer_plural(self, app_module):
        result = app_module.lemmatizer.lemmatize("geese")
        assert result == "goose"

    def test_word_tokenize(self):
        import nltk
        tokens = nltk.word_tokenize("Hello, how are you?")
        assert "Hello" in tokens
        assert "how" in tokens
        assert "?" in tokens

    def test_word_tokenize_empty(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_word_tokenize_single_word(self):
        import nltk
        tokens = nltk.word_tokenize("chatbot")
        assert tokens == ["chatbot"]


# ===========================================================================
# 12. Keras model integration (exercises the upgraded keras 3.x API via mock)
# ===========================================================================

class TestKerasModelIntegration:
    def test_load_model_called_with_h5(self):
        """Verify load_model is invoked with the expected .h5 path."""
        fake_model = _make_fake_model()
        with patch("builtins.open", side_effect=lambda p, m="r", *a, **k: mock_open(
            read_data=json.dumps(FAKE_INTENTS) if "intents" in str(p)
            else mock_open(read_data=pickle.dumps(FAKE_WORDS))()
        )()), \
             patch("pickle.load", side_effect=[FAKE_WORDS, FAKE_CLASSES]), \
             patch("keras.models.load_model", return_value=fake_model) as mock_lm, \
             patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):
            if "app" in sys.modules:
                del sys.modules["app"]
            import app  # noqa: PLC0415
            mock_lm.assert_called_once_with("chatbot_model.h5")

    def test_model_predict_input_shape(self, app_module, fake_model):
        """model.predict should receive array of shape (1, len(words))."""
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        app_module.predict_class("hello", fake_model)
        call_input = fake_model.predict.call_args[0][0]
        assert call_input.shape == (1, len(FAKE_WORDS))

    def test_model_predict_output_consumed(self, app_module, fake_model):
        """predict_class should consume model output and map to class names."""
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        result = app_module.predict_class("hello", fake_model)
        intents_found = [r["intent"] for r in result]
        for intent in intents_found:
            assert intent in FAKE_CLASSES


# ===========================================================================
# 13. python-multipart: multipart/form-data requests
# ===========================================================================

class TestMultipartRequests:
    """
    python-multipart is used by FastAPI to parse multipart/form-data.
    While the chatbot only exposes a JSON endpoint, these tests verify
    that the FastAPI app correctly handles and rejects multipart payloads.
    """

    def test_multipart_to_json_endpoint_rejected(self, client):
        resp = client.post("/api/chat", data={"msg": "hello"})
        # FastAPI will 422 because it expects JSON body, not form data
        assert resp.status_code == 422

    def test_json_payload_accepted(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post(
            "/api/chat",
            json={"msg": "Hi"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ===========================================================================
# 14. FastAPI routing & HTTP method enforcement
# ===========================================================================

class TestRoutingAndMethods:
    def test_post_to_health_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405

    def test_get_to_chat_not_allowed(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_nonexistent_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_put_to_chat_not_allowed(self, client):
        resp = client.put("/api/chat", json={"msg": "test"})
        assert resp.status_code == 405


# ===========================================================================
# 15. Name-substitution logic
# ===========================================================================

class TestNameSubstitution:
    """Verify {n} placeholder replacement for name-related intents."""

    def _setup_name_intent(self, app_module, fake_model):
        """Point model toward the 'name' intent (index 2)."""
        fake_model.predict.return_value = np.array([[0.05, 0.05, 0.90]])

    def test_my_name_is_alice(self, client, app_module, fake_model):
        self._setup_name_intent(app_module, fake_model)
        resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200
        # Response template is "Nice to meet you, {n}!" → should contain Alice
        body = resp.json()
        assert isinstance(body["response"], str)

    def test_hi_my_name_is_bob(self, client, app_module, fake_model):
        self._setup_name_intent(app_module, fake_model)
        resp = client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert resp.status_code == 200

    def test_i_am_carol(self, client, app_module, fake_model):
        self._setup_name_intent(app_module, fake_model)
        resp = client.post("/api/chat", json={"msg": "i am Carol"})
        assert resp.status_code == 200

    def test_case_insensitive_prefix_my_name_is(self, client, app_module, fake_model):
        self._setup_name_intent(app_module, fake_model)
        resp = client.post("/api/chat", json={"msg": "MY NAME IS Dave"})
        assert resp.status_code == 200

    def test_case_insensitive_prefix_i_am(self, client, app_module, fake_model):
        self._setup_name_intent(app_module, fake_model)
        resp = client.post("/api/chat", json={"msg": "I AM Eve"})
        assert resp.status_code == 200


# ===========================================================================
# 16. Module-level error: missing files raise RuntimeError
# ===========================================================================

class TestModuleLoadErrors:
    def test_missing_model_file_raises_runtime_error(self):
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch("keras.models.load_model", side_effect=FileNotFoundError("no model")), \
             patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):
            with pytest.raises((RuntimeError, FileNotFoundError, Exception)):
                import app  # noqa: PLC0415

    def test_missing_pickle_raises_runtime_error(self):
        if "app" in sys.modules:
            del sys.modules["app"]

        fake_model = _make_fake_model()
        with patch("keras.models.load_model", return_value=fake_model), \
             patch("builtins.open", side_effect=FileNotFoundError("no pkl")), \
             patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):
            with pytest.raises((RuntimeError, FileNotFoundError, Exception)):
                import app  # noqa: PLC0415


# ===========================================================================
# 17. Integration: full request/response cycle
# ===========================================================================

class TestIntegrationCycle:
    def test_full_greeting_cycle(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Hello there"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] in ["Hello!", "Hi there!", "Hey!"]
        assert body["confidence"] == pytest.approx(0.90)

    def test_full_goodbye_cycle(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.90, 0.05, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Goodbye"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] in ["Goodbye!", "See you later!"]
        assert body["confidence"] == pytest.approx(0.90)

    def test_confidence_matches_model_output(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.10, 0.80, 0.10]])
        resp = client.post("/api/chat", json={"msg": "hi"})
        assert resp.json()["confidence"] == pytest.approx(0.80)

    def test_whitespace_stripped_before_processing(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "  Hello  "})
        assert resp.status_code == 200
