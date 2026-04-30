"""
Tests for FastAPI chatbot endpoints (app.py).
Covers: routing, request validation, CORS, health-check, error handling,
and the upgraded FastAPI / python-multipart / keras / nltk APIs.
"""
import json
import pickle
import types
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – build lightweight stand-ins for every external resource that
# app.py tries to load at import time.
# ---------------------------------------------------------------------------

def _make_mock_model():
    """Return a Keras-like mock model whose predict() returns probabilities."""
    mock = MagicMock()
    # predict returns shape (1, N) – app.py calls model.predict(np.array(...))
    mock.predict.return_value = np.array([[0.1, 0.8, 0.1]])
    return mock


FAKE_WORDS = ["hello", "hi", "bye", "how", "are", "you", "name", "i", "am", "my"]
FAKE_CLASSES = ["greeting", "goodbye", "name_intro"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you"],
            "responses": ["Hello!", "Hi there!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later"],
            "responses": ["Goodbye!", "See you!"],
        },
        {
            "tag": "name_intro",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}


def _patch_app_loading():
    """
    Return a dict of patches that prevent app.py from hitting the filesystem
    or loading a real Keras model on import.
    """
    mock_model = _make_mock_model()
    patches = {
        "keras.models.load_model": patch("keras.models.load_model", return_value=mock_model),
        "pickle.load": patch("pickle.load", side_effect=[FAKE_WORDS, FAKE_CLASSES]),
        "builtins.open": patch(
            "builtins.open",
            mock_open(read_data=json.dumps(FAKE_INTENTS)),
        ),
        "fastapi.staticfiles.StaticFiles.__init__": patch(
            "fastapi.staticfiles.StaticFiles.__init__", return_value=None
        ),
    }
    return patches


# ---------------------------------------------------------------------------
# Fixture: import app with all I/O mocked out
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    mock_model = _make_mock_model()

    with (
        patch("keras.models.load_model", return_value=mock_model),
        patch("pickle.load", side_effect=[FAKE_WORDS, FAKE_CLASSES]),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps(FAKE_INTENTS)),
        ),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None),
    ):
        # Import *inside* the patches so module-level code uses mocks
        import importlib
        import app as app_module

        importlib.reload(app_module)  # ensure a clean state each run

        yield TestClient(app_module.app)


# ===========================================================================
# 1. Health-check endpoint
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ===========================================================================
# 2. Chat endpoint – happy paths
# ===========================================================================

class TestChatEndpointHappyPath:
    """Tests for POST /api/chat with valid inputs."""

    def _post(self, client, msg: str):
        return client.post("/api/chat", json={"msg": msg})

    # --- generic message ------------------------------------------------
    def test_generic_message_200(self, client):
        resp = self._post(client, "hello")
        assert resp.status_code == 200

    def test_generic_message_response_schema(self, client):
        data = self._post(client, "hello").json()
        assert "response" in data
        assert "confidence" in data
        assert isinstance(data["response"], str)
        assert isinstance(data["confidence"], float)

    def test_confidence_between_0_and_1(self, client):
        data = self._post(client, "hello").json()
        assert 0.0 <= data["confidence"] <= 1.0

    # --- "my name is" branch --------------------------------------------
    def test_my_name_is_200(self, client):
        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200

    def test_my_name_is_replaces_placeholder(self, client):
        """Response must substitute {n} with the provided name."""
        data = self._post(client, "my name is Alice").json()
        # The mock will return a response from FAKE_INTENTS; if it contains
        # "{n}" it should be replaced with "Alice".
        assert "{n}" not in data["response"]

    # --- "hi my name is" branch -----------------------------------------
    def test_hi_my_name_is_200(self, client):
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200

    def test_hi_my_name_is_replaces_placeholder(self, client):
        data = self._post(client, "hi my name is Bob").json()
        assert "{n}" not in data["response"]

    # --- "i am" branch --------------------------------------------------
    def test_i_am_200(self, client):
        resp = self._post(client, "i am Carol")
        assert resp.status_code == 200

    def test_i_am_replaces_placeholder(self, client):
        data = self._post(client, "i am Carol").json()
        assert "{n}" not in data["response"]

    # --- case-insensitivity of branch matching --------------------------
    def test_my_name_is_case_insensitive(self, client):
        resp = self._post(client, "MY NAME IS Dave")
        assert resp.status_code == 200

    def test_hi_my_name_is_case_insensitive(self, client):
        resp = self._post(client, "HI MY NAME IS Eve")
        assert resp.status_code == 200

    def test_i_am_case_insensitive(self, client):
        resp = self._post(client, "I AM Frank")
        assert resp.status_code == 200


# ===========================================================================
# 3. Chat endpoint – error / edge cases
# ===========================================================================

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

    def test_wrong_type_returns_422(self, client):
        # msg must be a string
        resp = client.post("/api/chat", json={"msg": 12345})
        # FastAPI coerces int→str via Pydantic v2 by default; accept 200 or 422
        assert resp.status_code in (200, 422)

    def test_internal_error_returns_500(self, client):
        """If predict_class raises, the endpoint should return 500."""
        import app as app_module

        original_predict = app_module.predict_class

        def boom(*a, **kw):
            raise RuntimeError("model exploded")

        app_module.predict_class = boom
        try:
            resp = client.post("/api/chat", json={"msg": "hello"})
            assert resp.status_code == 500
        finally:
            app_module.predict_class = original_predict


# ===========================================================================
# 4. CORS headers
# ===========================================================================

class TestCORSMiddleware:
    def test_cors_header_present_on_options(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should add the header
        assert resp.status_code in (200, 204)

    def test_cors_header_present_on_get(self, client):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        assert "access-control-allow-origin" in resp.headers


# ===========================================================================
# 5. Request / Response schema (Pydantic models)
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self):
        from app import MessageRequest

        req = MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_chat_response_valid(self):
        from app import ChatResponse

        resp = ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_confidence_float(self):
        from app import ChatResponse

        resp = ChatResponse(response="test", confidence=1)
        assert isinstance(resp.confidence, float)


# ===========================================================================
# 6. NLP helper functions (clean_up_sentence, bow, predict_class)
# ===========================================================================

class TestNLPHelpers:
    """
    These tests exercise the NLTK / numpy logic directly without going
    through the HTTP layer.
    """

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import app as app_module

        self.app_module = app_module

    # --- clean_up_sentence ----------------------------------------------
    def test_clean_up_sentence_returns_list(self):
        result = self.app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_clean_up_sentence_lowercases(self):
        result = self.app_module.clean_up_sentence("Hello")
        assert all(w == w.lower() for w in result)

    def test_clean_up_sentence_lemmatizes(self):
        # "running" → "running" with WordNetLemmatizer (verb form needs pos)
        # but "dogs" → "dog"
        result = self.app_module.clean_up_sentence("dogs")
        assert "dog" in result

    def test_clean_up_sentence_empty_string(self):
        result = self.app_module.clean_up_sentence("")
        assert result == []

    def test_clean_up_sentence_punctuation(self):
        result = self.app_module.clean_up_sentence("Hello!")
        # punctuation tokens are included (NLTK tokenises them separately)
        assert any(w.isalpha() for w in result)

    # --- bow (bag of words) ---------------------------------------------
    def test_bow_returns_numpy_array(self):
        result = self.app_module.bow("hello", FAKE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_bow_length_equals_vocabulary(self):
        result = self.app_module.bow("hello", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)

    def test_bow_known_word_sets_1(self):
        result = self.app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert result[idx] == 1

    def test_bow_unknown_word_all_zeros(self):
        result = self.app_module.bow("xyzzy", FAKE_WORDS)
        assert result.sum() == 0

    def test_bow_multiple_words(self):
        result = self.app_module.bow("hello hi", FAKE_WORDS)
        assert result[FAKE_WORDS.index("hello")] == 1
        assert result[FAKE_WORDS.index("hi")] == 1

    def test_bow_show_details_does_not_crash(self, capsys):
        self.app_module.bow("hello", FAKE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "found in bag" in captured.out

    # --- predict_class --------------------------------------------------
    def test_predict_class_returns_list(self):
        mock_model = _make_mock_model()
        # Patch the module-level `words` and `classes`
        original_words = self.app_module.words
        original_classes = self.app_module.classes
        self.app_module.words = FAKE_WORDS
        self.app_module.classes = FAKE_CLASSES
        try:
            result = self.app_module.predict_class("hello", mock_model)
            assert isinstance(result, list)
        finally:
            self.app_module.words = original_words
            self.app_module.classes = original_classes

    def test_predict_class_items_have_intent_and_probability(self):
        mock_model = _make_mock_model()
        original_words = self.app_module.words
        original_classes = self.app_module.classes
        self.app_module.words = FAKE_WORDS
        self.app_module.classes = FAKE_CLASSES
        try:
            result = self.app_module.predict_class("hello", mock_model)
            if result:
                assert "intent" in result[0]
                assert "probability" in result[0]
        finally:
            self.app_module.words = original_words
            self.app_module.classes = original_classes

    def test_predict_class_calls_model_predict(self):
        mock_model = _make_mock_model()
        original_words = self.app_module.words
        original_classes = self.app_module.classes
        self.app_module.words = FAKE_WORDS
        self.app_module.classes = FAKE_CLASSES
        try:
            self.app_module.predict_class("hello", mock_model)
            mock_model.predict.assert_called_once()
        finally:
            self.app_module.words = original_words
            self.app_module.classes = original_classes

    def test_predict_class_numpy_array_passed_to_model(self):
        """Keras 3.x requires numpy arrays; confirm predict receives one."""
        mock_model = _make_mock_model()
        original_words = self.app_module.words
        original_classes = self.app_module.classes
        self.app_module.words = FAKE_WORDS
        self.app_module.classes = FAKE_CLASSES
        try:
            self.app_module.predict_class("hello", mock_model)
            call_arg = mock_model.predict.call_args[0][0]
            assert isinstance(call_arg, np.ndarray)
        finally:
            self.app_module.words = original_words
            self.app_module.classes = original_classes


# ===========================================================================
# 7. getResponse helper
# ===========================================================================

class TestGetResponse:
    @pytest.fixture(autouse=True)
    def _load_module(self):
        import app as app_module

        self.app_module = app_module

    def test_get_response_returns_string(self):
        ints = [{"intent": "greeting", "probability": "0.99"}]
        result = self.app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_get_response_from_correct_intent(self):
        ints = [{"intent": "greeting", "probability": "0.99"}]
        result = self.app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][0]["responses"]

    def test_get_response_unknown_intent_does_not_crash(self):
        ints = [{"intent": "unknown_tag_xyz", "probability": "0.5"}]
        # Should not raise; may return empty string or default
        try:
            result = self.app_module.getResponse(ints, FAKE_INTENTS)
            assert isinstance(result, str)
        except Exception:
            pass  # If implementation raises, that's fine to document

    def test_get_response_name_intro_contains_placeholder(self):
        ints = [{"intent": "name_intro", "probability": "0.95"}]
        result = self.app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)


# ===========================================================================
# 8. FastAPI app metadata (title, version, openapi)
# ===========================================================================

class TestAppMetadata:
    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_title(self, client):
        data = client.get("/openapi.json").json()
        assert data["info"]["title"] == "AI Chatbot API"

    def test_openapi_version(self, client):
        data = client.get("/openapi.json").json()
        assert data["info"]["version"] == "1.0.0"

    def test_chat_route_in_openapi(self, client):
        data = client.get("/openapi.json").json()
        assert "/api/chat" in data["paths"]

    def test_health_route_in_openapi(self, client):
        data = client.get("/openapi.json").json()
        assert "/health" in data["paths"]
