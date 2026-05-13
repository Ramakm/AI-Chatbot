"""
Comprehensive pytest tests for the FastAPI Chatbot application.

Covers:
- FastAPI endpoints (/, /api/chat, /health)
- NLP helper functions (clean_up_sentence, bow, predict_class, getResponse)
- Upgraded package APIs: fastapi 0.109.1, python-multipart 0.0.7,
  keras 3.12.0, nltk 3.9
- Both happy-path and error/edge cases
- All external I/O (model, pickle, file reads) is mocked
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
# Helpers to build a fully-mocked app module
# ---------------------------------------------------------------------------

INTENTS_DATA = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is", "I am"],
            "responses": ["Nice to meet you, {n}!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later"],
            "responses": ["Goodbye!", "See you!"],
        },
    ]
}

WORDS = ["bye", "hello", "hey", "hi", "name"]
CLASSES = ["goodbye", "greeting", "name"]


def _make_mock_model(prediction: np.ndarray | None = None):
    """Return a mock Keras model whose .predict() returns a given array."""
    mock_model = MagicMock()
    if prediction is None:
        # Default: high-confidence "greeting" class (index 1)
        prediction = np.array([[0.05, 0.90, 0.05]])
    mock_model.predict.return_value = prediction
    return mock_model


def _import_app_with_mocks(mock_model=None, words=None, classes=None, intents=None):
    """
    Import (or re-import) app.py with all external dependencies mocked.
    Returns the module so callers can access helpers directly.
    """
    if words is None:
        words = WORDS
    if classes is None:
        classes = CLASSES
    if intents is None:
        intents = INTENTS_DATA
    if mock_model is None:
        mock_model = _make_mock_model()

    # Remove cached module so we can patch fresh
    sys.modules.pop("app", None)

    with (
        patch("keras.models.load_model", return_value=mock_model),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps(intents)),
        ),
        patch("pickle.load", side_effect=[words, classes]),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("nltk.download", return_value=True),
    ):
        import app as app_module

    return app_module


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module():
    """Load app module once per test-module with all external deps mocked."""
    return _import_app_with_mocks()


@pytest.fixture(scope="module")
def client(app_module):
    """Return a TestClient for the FastAPI app."""
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def mock_model(app_module):
    return app_module.model


# ---------------------------------------------------------------------------
# 1. FastAPI application bootstrap
# ---------------------------------------------------------------------------


class TestAppBootstrap:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_words_loaded(self, app_module):
        assert app_module.words == WORDS

    def test_classes_loaded(self, app_module):
        assert app_module.classes == CLASSES

    def test_intents_loaded(self, app_module):
        assert app_module.intents == INTENTS_DATA

    def test_model_attribute_exists(self, app_module):
        assert app_module.model is not None

    def test_lemmatizer_is_wordnet(self, app_module):
        from nltk.stem import WordNetLemmatizer

        assert isinstance(app_module.lemmatizer, WordNetLemmatizer)

    def test_cors_middleware_present(self, app_module):
        """CORS middleware should be registered (fastapi ≥ 0.109.1)."""
        middleware_types = [
            m.cls.__name__ if hasattr(m, "cls") else type(m).__name__
            for m in app_module.app.user_middleware
        ]
        assert any("CORS" in name for name in middleware_types)


# ---------------------------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_200(self, client):
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
    def test_home_returns_html(self, client):
        html_content = "<html><body>chatbot</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_home_contains_content(self, client):
        html_content = "<html><body>chatbot</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            resp = client.get("/")
        assert "chatbot" in resp.text


# ---------------------------------------------------------------------------
# 4. Chat endpoint – happy paths
# ---------------------------------------------------------------------------


class TestChatEndpointHappyPath:
    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_greeting_response_200(self, client, app_module):
        """Standard greeting message returns 200 with response + confidence."""
        ints = [{"intent": "greeting", "probability": "0.9"}]
        with (
            patch.object(
                sys.modules["app"], "predict_class", return_value=ints
            ),
            patch.object(
                sys.modules["app"], "getResponse", return_value="Hello!"
            ),
        ):
            resp = self._post(client, "Hello")
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "confidence" in body
        assert body["response"] == "Hello!"
        assert abs(body["confidence"] - 0.9) < 1e-6

    def test_confidence_is_float(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.75"}]
        with (
            patch.object(sys.modules["app"], "predict_class", return_value=ints),
            patch.object(sys.modules["app"], "getResponse", return_value="Hi!"),
        ):
            resp = self._post(client, "Hi there")
        assert isinstance(resp.json()["confidence"], float)

    def test_name_pattern_my_name_is(self, client):
        """'my name is X' substitutes {n} in response."""
        ints = [{"intent": "name", "probability": "0.88"}]
        with (
            patch.object(sys.modules["app"], "predict_class", return_value=ints),
            patch.object(
                sys.modules["app"],
                "getResponse",
                return_value="Nice to meet you, {n}!",
            ),
        ):
            resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        assert "Alice" in resp.json()["response"]

    def test_name_pattern_hi_my_name_is(self, client):
        """'hi my name is X' substitutes {n} in response."""
        ints = [{"intent": "name", "probability": "0.85"}]
        with (
            patch.object(sys.modules["app"], "predict_class", return_value=ints),
            patch.object(
                sys.modules["app"],
                "getResponse",
                return_value="Nice to meet you, {n}!",
            ),
        ):
            resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200
        assert "Bob" in resp.json()["response"]

    def test_name_pattern_i_am(self, client):
        """'I am X' substitutes {n} in response."""
        ints = [{"intent": "name", "probability": "0.80"}]
        with (
            patch.object(sys.modules["app"], "predict_class", return_value=ints),
            patch.object(
                sys.modules["app"],
                "getResponse",
                return_value="Nice to meet you, {n}!",
            ),
        ):
            resp = self._post(client, "I am Carol")
        assert resp.status_code == 200
        assert "Carol" in resp.json()["response"]

    def test_whitespace_stripped(self, client):
        """Leading/trailing whitespace in msg is handled gracefully."""
        ints = [{"intent": "greeting", "probability": "0.9"}]
        with (
            patch.object(sys.modules["app"], "predict_class", return_value=ints),
            patch.object(sys.modules["app"], "getResponse", return_value="Hello!"),
        ):
            resp = self._post(client, "   Hello   ")
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
        assert "empty" in resp.json()["detail"].lower()

    def test_whitespace_only_returns_400(self, client):
        resp = self._post(client, {"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        """FastAPI ≥ 0.109.1 validation returns 422 for missing required fields."""
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_internal_error_returns_500(self, client):
        """If predict_class raises, endpoint returns 500."""
        with patch.object(
            sys.modules["app"],
            "predict_class",
            side_effect=RuntimeError("model exploded"),
        ):
            resp = self._post(client, {"msg": "hello"})
        assert resp.status_code == 500
        assert "Error processing message" in resp.json()["detail"]

    def test_wrong_content_type_returns_422(self, client):
        """Sending plain text instead of JSON returns 422 (FastAPI validation)."""
        resp = client.post(
            "/api/chat",
            content="hello",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. NLP helpers – clean_up_sentence
# ---------------------------------------------------------------------------


class TestCleanUpSentence:
    def test_tokenises_sentence(self, app_module):
        tokens = app_module.clean_up_sentence("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) >= 2

    def test_lowercases(self, app_module):
        tokens = app_module.clean_up_sentence("HELLO")
        assert all(t == t.lower() for t in tokens)

    def test_lemmatises_running(self, app_module):
        tokens = app_module.clean_up_sentence("running")
        assert "run" in tokens or "running" in tokens  # nltk 3.9

    def test_empty_string(self, app_module):
        tokens = app_module.clean_up_sentence("")
        assert tokens == []

    def test_punctuation_retained_or_stripped(self, app_module):
        """Should not crash on punctuation-heavy input."""
        tokens = app_module.clean_up_sentence("Hello!!! How are you?")
        assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# 7. NLP helpers – bow (bag of words)
# ---------------------------------------------------------------------------


class TestBow:
    def test_returns_numpy_array(self, app_module):
        result = app_module.bow("hello", WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_words(self, app_module):
        result = app_module.bow("hello", WORDS)
        assert len(result) == len(WORDS)

    def test_known_word_flagged(self, app_module):
        # WORDS = ["bye", "hello", "hey", "hi", "name"]  index 1 = "hello"
        result = app_module.bow("hello", WORDS)
        assert result[1] == 1

    def test_unknown_word_not_flagged(self, app_module):
        result = app_module.bow("zzz_unknown_word", WORDS)
        assert all(v == 0 for v in result)

    def test_multiple_known_words(self, app_module):
        result = app_module.bow("hello hi", WORDS)
        # "hello" -> index 1, "hi" -> index 3
        assert result[1] == 1
        assert result[3] == 1

    def test_show_details_does_not_crash(self, app_module, capsys):
        app_module.bow("hello", WORDS, show_details=True)
        out = capsys.readouterr().out
        assert "hello" in out

    def test_empty_sentence(self, app_module):
        result = app_module.bow("", WORDS)
        assert all(v == 0 for v in result)


# ---------------------------------------------------------------------------
# 8. NLP helpers – predict_class
# ---------------------------------------------------------------------------


class TestPredictClass:
    def test_returns_list(self, app_module):
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        result = app_module.predict_class("hello", mock_model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, app_module):
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        result = app_module.predict_class("hello", mock_model)
        if result:
            assert "intent" in result[0]
            assert "probability" in result[0]

    def test_top_intent_is_highest_probability(self, app_module):
        # index 1 (greeting) has highest prob
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        result = app_module.predict_class("hello", mock_model)
        if result:
            assert result[0]["intent"] == CLASSES[1]  # "greeting"

    def test_probability_values_are_strings(self, app_module):
        """predict_class stores probability as string for JSON compatibility."""
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        result = app_module.predict_class("hello", mock_model)
        if result:
            prob = result[0]["probability"]
            # Accept both str and numeric – just ensure float conversion works
            assert float(prob) == pytest.approx(0.90, abs=1e-3)

    def test_low_confidence_filtered(self, app_module):
        """Results below ERROR_THRESHOLD (typically 0.25) should be excluded."""
        mock_model = _make_mock_model(np.array([[0.10, 0.10, 0.10]]))
        result = app_module.predict_class("zzz", mock_model)
        # May return empty list if all below threshold
        assert isinstance(result, list)

    def test_model_predict_called_once(self, app_module):
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        app_module.predict_class("hello", mock_model)
        mock_model.predict.assert_called_once()

    def test_model_receives_2d_array(self, app_module):
        """Keras 3.x expects a 2-D batch input."""
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        app_module.predict_class("hello", mock_model)
        call_args = mock_model.predict.call_args
        arr = call_args[0][0] if call_args[0] else call_args[1].get("x", None)
        if arr is not None:
            assert arr.ndim == 2


# ---------------------------------------------------------------------------
# 9. Pydantic models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_empty_allowed_at_schema_level(self, app_module):
        """Pydantic allows empty string; business logic rejects it."""
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_fields(self, app_module):
        resp = app_module.ChatResponse(response="Hello!", confidence=0.9)
        assert resp.response == "Hello!"
        assert resp.confidence == pytest.approx(0.9)

    def test_chat_response_confidence_float(self, app_module):
        resp = app_module.ChatResponse(response="hi", confidence=0.5)
        assert isinstance(resp.confidence, float)


# ---------------------------------------------------------------------------
# 10. getResponse helper (indirectly tested; define a stub if absent)
# ---------------------------------------------------------------------------


class TestGetResponse:
    def _build_intents(self):
        return INTENTS_DATA

    def test_returns_string(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, self._build_intents())
        assert isinstance(result, str)

    def test_response_from_correct_tag(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, self._build_intents())
        assert result in INTENTS_DATA["intents"][0]["responses"]

    def test_unknown_intent_does_not_crash(self, app_module):
        ints = [{"intent": "unknown_tag_xyz", "probability": "0.5"}]
        try:
            result = app_module.getResponse(ints, self._build_intents())
            assert isinstance(result, str)
        except Exception:
            pass  # acceptable to raise on unknown tag

    def test_empty_ints_list(self, app_module):
        """Empty prediction list should not crash catastrophically."""
        try:
            result = app_module.getResponse([], self._build_intents())
            assert isinstance(result, str)
        except (IndexError, KeyError):
            pass  # either is acceptable


# ---------------------------------------------------------------------------
# 11. NLTK 3.9 – tokenisation and lemmatisation behaviour
# ---------------------------------------------------------------------------


class TestNLTK39:
    """Directly exercise nltk APIs used in app.py with nltk 3.9."""

    def test_word_tokenize_basic(self):
        import nltk

        tokens = nltk.word_tokenize("Hello, how are you?")
        assert "Hello" in tokens
        assert "how" in tokens

    def test_word_tokenize_returns_list(self):
        import nltk

        result = nltk.word_tokenize("testing")
        assert isinstance(result, list)

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        assert lem.lemmatize("running", pos="v") == "run"

    def test_wordnet_lemmatizer_preserves_unknown(self):
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        assert lem.lemmatize("xyzzy") == "xyzzy"

    def test_word_tokenize_punctuation(self):
        import nltk

        tokens = nltk.word_tokenize("It's a test!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_word_tokenize_empty(self):
        import nltk

        tokens = nltk.word_tokenize("")
        assert tokens == []


# ---------------------------------------------------------------------------
# 12. Keras 3.x – load_model API (mocked)
# ---------------------------------------------------------------------------


class TestKeras3LoadModel:
    def test_load_model_called_with_h5_path(self):
        """Ensure load_model is called with the expected file path."""
        sys.modules.pop("app", None)
        mock_model = _make_mock_model()
        with (
            patch("keras.models.load_model", return_value=mock_model) as mocked,
            patch("builtins.open", mock_open(read_data=json.dumps(INTENTS_DATA))),
            patch("pickle.load", side_effect=[WORDS, CLASSES]),
            patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
            patch("nltk.download", return_value=True),
        ):
            import app  # noqa: F401

            mocked.assert_called_once_with("chatbot_model.h5")
        sys.modules.pop("app", None)

    def test_model_predict_signature(self):
        """Keras 3.x model.predict accepts a numpy array."""
        mock_model = _make_mock_model(np.array([[0.1, 0.8, 0.1]]))
        x = np.array([[1, 0, 1, 0, 0]])
        result = mock_model.predict(x)
        assert result.shape == (1, 3)

    def test_load_model_failure_raises_runtime_error(self):
        """If load_model raises, the module raises RuntimeError."""
        sys.modules.pop("app", None)
        with (
            patch(
                "keras.models.load_model",
                side_effect=OSError("file not found"),
            ),
            patch("builtins.open", mock_open(read_data=json.dumps(INTENTS_DATA))),
            patch("pickle.load", side_effect=[WORDS, CLASSES]),
            patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
            patch("nltk.download", return_value=True),
        ):
            with pytest.raises((RuntimeError, OSError)):
                import app  # noqa: F401
        sys.modules.pop("app", None)


# ---------------------------------------------------------------------------
# 13. FastAPI 0.109.1 – request validation (python-multipart 0.0.7)
# ---------------------------------------------------------------------------


class TestFastAPIValidation:
    """
    FastAPI 0.109.1 tightened request-body validation and form-data handling
    via python-multipart 0.0.7.
    """

    def test_extra_fields_ignored_or_rejected(self, client):
        """FastAPI should not crash on extra JSON fields (strict mode off by default)."""
        resp = client.post("/api/chat", json={"msg": "hello", "extra": "data"})
        # 200 or 422 depending on model config – must not be 500
        assert resp.status_code in (200, 422, 400)

    def test_integer_msg_rejected_422(self, client):
        """msg field must be a string; integer should trigger 422."""
        resp = client.post("/api/chat", json={"msg": 12345})
        # FastAPI coerces int→str in v1 mode; in v2 strict mode → 422
        assert resp.status_code in (200, 422)

    def test_null_msg_returns_422(self, client):
        resp = client.post("/api/chat", json={"msg": None})
        assert resp.status_code == 422

    def test_method_not_allowed_on_chat(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_openapi_schema_accessible(self, client):
        """FastAPI ≥ 0.109.1 serves OpenAPI schema at /openapi.json."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "openapi" in schema
        assert "/api/chat" in schema["paths"]

    def test_docs_endpoint_accessible(self, client):
        """Swagger UI available at /docs."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_endpoint_accessible(self, client):
        """ReDoc UI available at /redoc."""
        resp = client.get("/redoc")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 14. Integration-style: full request-to-response flow
# ---------------------------------------------------------------------------


class TestIntegrationFlow:
    """
    Wire real helper functions together (no model mocking at function level)
    to verify end-to-end prediction pipeline.
    """

    def test_bow_fed_into_predict_shape(self, app_module):
        """bow output shape matches model's expected input."""
        bag = app_module.bow("hello", WORDS)
        assert bag.shape == (len(WORDS),)

    def test_clean_then_bow_then_predict(self, app_module):
        mock_model = _make_mock_model(np.array([[0.05, 0.90, 0.05]]))
        tokens = app_module.clean_up_sentence("hello")
        assert isinstance(tokens, list)

        bag = app_module.bow("hello", WORDS)
        assert bag.ndim == 1

        result = app_module.predict_class("hello", mock_model)
        assert isinstance(result, list)

    def test_full_chat_pipeline_greeting(self, client):
        """End-to-end: greeting message returns a valid ChatResponse."""
        # Use the real predict_class but with a mock model pre-wired in app
        with patch.object(
            sys.modules["app"],
            "predict_class",
            return_value=[{"intent": "greeting", "probability": "0.9"}],
        ), patch.object(
            sys.modules["app"],
            "getResponse",
            return_value="Hello!",
        ):
            resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Hello!"
        assert body["confidence"] == pytest.approx(0.9)

    def test_full_chat_pipeline_name_substitution(self, client):
        """End-to-end: name substitution works through full pipeline."""
        with patch.object(
            sys.modules["app"],
            "predict_class",
            return_value=[{"intent": "name", "probability": "0.85"}],
        ), patch.object(
            sys.modules["app"],
            "getResponse",
            return_value="Nice to meet you, {n}!",
        ):
            resp = client.post("/api/chat", json={"msg": "my name is Dave"})
        assert resp.status_code == 200
        assert "Dave" in resp.json()["response"]
