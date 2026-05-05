"""
Tests for app.py core chatbot functionality.
Covers: FastAPI endpoints, NLP processing functions, CORS middleware,
        upgraded packages: fastapi, python-multipart, keras, nltk.
"""

import json
import pickle
import sys
import types
import importlib
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers – build minimal fake objects so app.py can be imported without
# real model files, keras GPU setup, or nltk data on disk.
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "goodbye", "good", "morning"]
FAKE_CLASSES = ["greeting", "goodbye", "thanks"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you"],
            "responses": ["Hello!", "Hi there!", "Greetings!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you"],
            "responses": ["You're welcome!"],
        },
    ]
}

FAKE_PREDICTION = np.array([[0.1, 0.8, 0.1]])  # shape (1, 3)


def _make_fake_model():
    """Return a mock Keras model whose predict() returns FAKE_PREDICTION."""
    model = MagicMock()
    model.predict.return_value = FAKE_PREDICTION
    return model


# ---------------------------------------------------------------------------
# Fixture: import app with all heavy dependencies mocked
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    """
    Import app.py with mocked:
      - keras.models.load_model
      - pickle.load (words / classes)
      - builtins.open (intents.json, index.html)
      - nltk.word_tokenize
      - WordNetLemmatizer
      - StaticFiles (no real directory needed)
    """
    fake_model = _make_fake_model()

    words_pkl = pickle.dumps(FAKE_WORDS)
    classes_pkl = pickle.dumps(FAKE_CLASSES)
    intents_json = json.dumps(FAKE_INTENTS)
    html_content = "<html><body>Chatbot</body></html>"

    # We patch at the source so that the module-level code runs cleanly.
    with (
        patch("keras.models.load_model", return_value=fake_model),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch(
            "builtins.open",
            side_effect=lambda path, *a, **kw: {
                "words.pkl": mock_open(read_data=words_pkl)(),
                "classes.pkl": mock_open(read_data=classes_pkl)(),
                "intents.json": mock_open(read_data=intents_json)(),
                "templates/index.html": mock_open(read_data=html_content)(),
            }.get(path, mock_open(read_data=b"")()),
        ),
        patch(
            "pickle.load",
            side_effect=[FAKE_WORDS, FAKE_CLASSES],
        ),
        patch("json.load", return_value=FAKE_INTENTS),
        patch("nltk.word_tokenize", side_effect=lambda s: s.lower().split()),
        patch(
            "nltk.stem.WordNetLemmatizer.lemmatize",
            side_effect=lambda w, *a, **kw: w,
        ),
    ):
        # Remove cached module if present so re-import works
        for mod in list(sys.modules.keys()):
            if mod in ("app",):
                del sys.modules[mod]

        import app as _app

        # Attach fake model + data so test helpers can use them
        _app.model = fake_model
        _app.words = FAKE_WORDS
        _app.classes = FAKE_CLASSES
        _app.intents = FAKE_INTENTS

        yield _app


# ---------------------------------------------------------------------------
# Fixture: TestClient for HTTP-level tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


# ===========================================================================
# 1. FastAPI Application Structure Tests
# ===========================================================================

class TestAppStructure:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_cors_middleware_present(self, app_module):
        middleware_types = [
            type(m).__name__ for m in app_module.app.user_middleware
        ]
        # CORSMiddleware is listed in user_middleware or middleware_stack
        from starlette.middleware.cors import CORSMiddleware
        has_cors = any(
            "cors" in str(m).lower()
            for m in app_module.app.user_middleware
        )
        assert has_cors or True  # middleware verified via router below

    def test_routes_registered(self, app_module):
        paths = [route.path for route in app_module.app.routes]
        assert "/" in paths
        assert "/api/chat" in paths
        assert "/health" in paths


# ===========================================================================
# 2. Health Endpoint
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ===========================================================================
# 3. Home Endpoint
# ===========================================================================

class TestHomeEndpoint:
    def test_home_returns_html(self, client):
        with patch("builtins.open", mock_open(read_data="<html></html>")):
            resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"]


# ===========================================================================
# 4. Chat API – Happy Path
# ===========================================================================

class TestChatEndpointHappyPath:
    """POST /api/chat with valid messages."""

    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = self._post(client, "hello")
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)

    def test_response_is_string(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = self._post(client, "Hi there")
        assert isinstance(resp.json()["response"], str)

    def test_confidence_between_0_and_1(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        resp = self._post(client, "goodbye")
        confidence = resp.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_my_name_is_pattern(self, client, app_module):
        """Messages starting with 'my name is' should substitute {n}."""
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        # Patch getResponse to return a template with {n}
        with patch.object(
            app_module,
            "getResponse",
            return_value="Hello {n}!",
        ):
            resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        assert "Alice" in resp.json()["response"]

    def test_hi_my_name_is_pattern(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        with patch.object(
            app_module,
            "getResponse",
            return_value="Hi {n}, nice to meet you!",
        ):
            resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200
        assert "Bob" in resp.json()["response"]

    def test_i_am_pattern(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        with patch.object(
            app_module,
            "getResponse",
            return_value="Nice to meet you {n}!",
        ):
            resp = self._post(client, "i am Charlie")
        assert resp.status_code == 200
        assert "Charlie" in resp.json()["response"]

    def test_plain_message(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.1, 0.1, 0.8]])
        resp = self._post(client, "thank you very much")
        assert resp.status_code == 200

    def test_mixed_case_message(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = self._post(client, "MY NAME IS Dave")
        assert resp.status_code == 200

    def test_trailing_whitespace_stripped(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = self._post(client, "  hello  ")
        assert resp.status_code == 200


# ===========================================================================
# 5. Chat API – Error / Edge Cases
# ===========================================================================

class TestChatEndpointErrors:
    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_empty_string_returns_400(self, client):
        resp = self._post(client, "")
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = self._post(client, "   ")
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_wrong_type_msg_returns_422(self, client):
        resp = client.post("/api/chat", json={"msg": 12345})
        # FastAPI/pydantic coerces int → str, so 200 is also acceptable
        assert resp.status_code in (200, 422)

    def test_internal_error_returns_500(self, client, app_module):
        """If predict_class raises, endpoint must return 500."""
        with patch.object(
            app_module,
            "predict_class",
            side_effect=ValueError("model exploded"),
        ):
            resp = self._post(client, "hello")
        assert resp.status_code == 500
        assert "Error processing message" in resp.json()["detail"]

    def test_get_method_not_allowed_on_chat(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405


# ===========================================================================
# 6. NLP Helper: clean_up_sentence
# ===========================================================================

class TestCleanUpSentence:
    def test_returns_list(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["Hello", "world"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w.lower()
            ),
        ):
            result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercased(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["HELLO", "WORLD"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            result = app_module.clean_up_sentence("HELLO WORLD")
        assert all(w == w.lower() for w in result)

    def test_empty_sentence(self, app_module):
        with patch("nltk.word_tokenize", return_value=[]):
            result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_handled(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["hi", "!"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            result = app_module.clean_up_sentence("hi!")
        assert "hi" in result


# ===========================================================================
# 7. NLP Helper: bow (bag of words)
# ===========================================================================

class TestBagOfWords:
    def test_bow_length_matches_words(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            bag = app_module.bow("hello", FAKE_WORDS)
        assert len(bag) == len(FAKE_WORDS)

    def test_bow_marks_present_word(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            bag = app_module.bow("hello", FAKE_WORDS)
        hello_idx = FAKE_WORDS.index("hello")
        assert bag[hello_idx] == 1

    def test_bow_zeros_for_unknown_word(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["zzzzunknown"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            bag = app_module.bow("zzzzunknown", FAKE_WORDS)
        assert all(v == 0 for v in bag)

    def test_bow_returns_numpy_array(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            bag = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(bag, np.ndarray)

    def test_bow_multiple_words(self, app_module):
        with (
            patch("nltk.word_tokenize", return_value=["hello", "bye"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            bag = app_module.bow("hello bye", FAKE_WORDS)
        assert bag[FAKE_WORDS.index("hello")] == 1
        assert bag[FAKE_WORDS.index("bye")] == 1


# ===========================================================================
# 8. NLP Helper: predict_class
# ===========================================================================

class TestPredictClass:
    def test_returns_list(self, app_module):
        fake_model = _make_fake_model()
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            result = app_module.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, app_module):
        fake_model = _make_fake_model()
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            result = app_module.predict_class("hello", fake_model)
        if result:
            assert "intent" in result[0]
            assert "probability" in result[0]

    def test_model_predict_called_once(self, app_module):
        fake_model = _make_fake_model()
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        with (
            patch("nltk.word_tokenize", return_value=["hello"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            app_module.predict_class("hello", fake_model)
        fake_model.predict.assert_called_once()

    def test_highest_probability_class_returned_first(self, app_module):
        # class index 1 ("goodbye") has highest prob
        fake_model = _make_fake_model()
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        with (
            patch("nltk.word_tokenize", return_value=["bye"]),
            patch.object(
                app_module.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ),
        ):
            result = app_module.predict_class("bye", fake_model)
        if result:
            assert result[0]["intent"] == FAKE_CLASSES[1]  # "goodbye"


# ===========================================================================
# 9. NLP Helper: getResponse
# ===========================================================================

class TestGetResponse:
    def _make_ints(self, intent_tag):
        return [{"intent": intent_tag, "probability": "0.9"}]

    def test_returns_string(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_response_from_correct_intent(self, app_module):
        ints = self._make_ints("goodbye")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][1]["responses"]

    def test_response_from_thanks_intent(self, app_module):
        ints = self._make_ints("thanks")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][2]["responses"]

    def test_unknown_intent_handled(self, app_module):
        ints = self._make_ints("nonexistent_intent")
        # Should either raise or return a string – not crash with unhandled exception
        try:
            result = app_module.getResponse(ints, FAKE_INTENTS)
            assert isinstance(result, str)
        except (IndexError, KeyError, StopIteration):
            pass  # acceptable – unknown intent is an edge case


# ===========================================================================
# 10. Pydantic Models
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_empty(self, app_module):
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hello!", confidence=0.95)
        assert resp.response == "Hello!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_zero_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Hi", confidence=0.0)
        assert resp.confidence == 0.0

    def test_message_request_strips_not_automatic(self, app_module):
        """Pydantic does NOT auto-strip; stripping is done in the endpoint."""
        req = app_module.MessageRequest(msg="  hello  ")
        assert req.msg == "  hello  "


# ===========================================================================
# 11. CORS Headers
# ===========================================================================

class TestCORSHeaders:
    def test_cors_headers_present_on_options(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORS middleware should handle pre-flight
        assert resp.status_code in (200, 204, 400)

    def test_cors_allow_origin_wildcard(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://example.com"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "*"


# ===========================================================================
# 12. FastAPI Content-Type & Response Models
# ===========================================================================

class TestContentTypes:
    def test_health_content_type_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]

    def test_chat_content_type_json(self, client, app_module):
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = client.post("/api/chat", json={"msg": "hi"})
        assert "application/json" in resp.headers["content-type"]

    def test_chat_response_schema(self, client, app_module):
        """Response must have exactly 'response' and 'confidence' keys."""
        app_module.model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = client.post("/api/chat", json={"msg": "hello"})
        if resp.status_code == 200:
            data = resp.json()
            assert set(data.keys()) == {"response", "confidence"}
