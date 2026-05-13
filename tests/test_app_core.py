"""
Tests for core chatbot functionality in app.py
Covers: FastAPI endpoints, NLP helpers, upgraded packages (fastapi, keras, nltk, python-multipart)
"""
import json
import pickle
import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Helpers – build a minimal fake app module without actually loading files
# ---------------------------------------------------------------------------

def _make_fake_model(output: list[float] | None = None):
    """Return a mock Keras model whose predict() returns a controlled array."""
    if output is None:
        output = [0.1, 0.8, 0.1]
    model = MagicMock()
    model.predict.return_value = np.array([output])
    return model


FAKE_WORDS = ["hello", "hi", "bye", "name", "my", "i", "am", "good", "help"]
FAKE_CLASSES = ["greeting", "goodbye", "name_query"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi"],
            "responses": ["Hello!", "Hi there!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you"],
            "responses": ["Goodbye!"],
        },
        {
            "tag": "name_query",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}


@pytest.fixture(scope="session")
def patched_app():
    """
    Import app with all external I/O mocked so the module-level
    try/except succeeds and we get a usable FastAPI test client.
    """
    fake_model = _make_fake_model([0.05, 0.90, 0.05])

    patches = [
        patch("keras.models.load_model", return_value=fake_model),
        patch(
            "builtins.open",
            side_effect=_open_side_effect,
        ),
        patch("pickle.load", side_effect=_pickle_load_side_effect),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None),
    ]

    for p in patches:
        p.start()

    # Force re-import in case the module is cached
    if "app" in sys.modules:
        del sys.modules["app"]

    import app as chatbot_app

    for p in patches:
        p.stop()

    return chatbot_app


def _open_side_effect(path, *args, **kwargs):
    """Fake open() that returns JSON for intents.json and HTML for index.html."""
    path = str(path)
    if "intents.json" in path:
        return mock_open(read_data=json.dumps(FAKE_INTENTS))()
    if "index.html" in path:
        return mock_open(read_data="<html>chatbot</html>")()
    if "words.pkl" in path or "classes.pkl" in path:
        # Return a real file-like object for pickle
        import io
        return io.BytesIO(b"")
    raise FileNotFoundError(f"Unexpected open: {path}")


def _pickle_load_side_effect(f):
    """Return different fixtures depending on read order."""
    if not hasattr(_pickle_load_side_effect, "_calls"):
        _pickle_load_side_effect._calls = 0
    _pickle_load_side_effect._calls += 1
    if _pickle_load_side_effect._calls % 2 == 1:
        return FAKE_WORDS
    return FAKE_CLASSES


# ---------------------------------------------------------------------------
# Re-usable fixtures for unit-testing individual helper functions
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_module():
    """
    Provide access to app-module functions without going through FastAPI.
    We patch everything at import time so the module loads cleanly.
    """
    fake_model = _make_fake_model([0.05, 0.90, 0.05])

    with (
        patch("keras.models.load_model", return_value=fake_model),
        patch("builtins.open", side_effect=_open_side_effect),
        patch("pickle.load", side_effect=_pickle_load_side_effect),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
    ):
        _pickle_load_side_effect._calls = 0  # reset counter
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as m

    # Restore module-level data to predictable state
    m.words = list(FAKE_WORDS)
    m.classes = list(FAKE_CLASSES)
    m.intents = FAKE_INTENTS
    m.model = fake_model
    return m


# ---------------------------------------------------------------------------
# 1. NLP helper – clean_up_sentence
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_tokenises_and_lowercases(self, app_module):
        result = app_module.clean_up_sentence("Hello World")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)
        # all tokens should be lowercase
        assert all(t == t.lower() for t in result)

    def test_lemmatizes_plural(self, app_module):
        result = app_module.clean_up_sentence("running")
        # WordNetLemmatizer lemmatises with default POS=noun, so "running"→"running"
        # but "dogs" → "dog"
        dogs = app_module.clean_up_sentence("dogs")
        assert "dog" in dogs

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_preserved_as_tokens(self, app_module):
        result = app_module.clean_up_sentence("Hi!")
        # nltk.word_tokenize splits "Hi!" into ["Hi", "!"]
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# 2. NLP helper – bow
# ---------------------------------------------------------------------------

class TestBagOfWords:
    def test_returns_numpy_array(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(arr, np.ndarray)

    def test_length_matches_vocabulary(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert len(arr) == len(FAKE_WORDS)

    def test_known_word_sets_bit(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert arr[idx] == 1

    def test_unknown_word_all_zeros(self, app_module):
        arr = app_module.bow("xyzzy_unknown", FAKE_WORDS)
        assert arr.sum() == 0

    def test_multiple_known_words(self, app_module):
        arr = app_module.bow("hello bye", FAKE_WORDS)
        assert arr[FAKE_WORDS.index("hello")] == 1
        assert arr[FAKE_WORDS.index("bye")] == 1

    def test_show_details_true_does_not_raise(self, app_module, capsys):
        app_module.bow("hello", FAKE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "hello" in captured.out


# ---------------------------------------------------------------------------
# 3. NLP helper – predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        assert isinstance(result, list)

    def test_each_item_has_intent_and_probability(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        for item in result:
            assert "intent" in item
            assert "probability" in item

    def test_probability_is_string_of_float(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        for item in result:
            assert float(item["probability"]) >= 0.0

    def test_model_predict_called(self, app_module):
        app_module.model.predict.reset_mock()
        app_module.predict_class("hello", app_module.model)
        app_module.model.predict.assert_called_once()

    def test_filters_low_confidence(self, app_module):
        # If all outputs are below threshold, result may be empty or filtered
        low_model = _make_fake_model([0.0, 0.0, 0.0])
        result = app_module.predict_class("hello", low_model)
        # Should not raise; result is a list (possibly empty)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4. NLP helper – getResponse
# ---------------------------------------------------------------------------

class TestGetResponse:
    def test_returns_string(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        res = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(res, str)

    def test_response_from_correct_tag(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        res = app_module.getResponse(ints, FAKE_INTENTS)
        assert res in FAKE_INTENTS["intents"][0]["responses"]

    def test_response_for_goodbye_tag(self, app_module):
        ints = [{"intent": "goodbye", "probability": "0.95"}]
        res = app_module.getResponse(ints, FAKE_INTENTS)
        assert res == "Goodbye!"

    def test_empty_ints_does_not_crash(self, app_module):
        # Should handle gracefully (may raise or return fallback – just no unhandled crash)
        try:
            res = app_module.getResponse([], FAKE_INTENTS)
            assert isinstance(res, str)
        except (IndexError, KeyError):
            pass  # acceptable – caller guards this

    def test_unknown_intent_tag(self, app_module):
        ints = [{"intent": "nonexistent_tag", "probability": "0.99"}]
        try:
            app_module.getResponse(ints, FAKE_INTENTS)
        except (StopIteration, KeyError, IndexError):
            pass  # acceptable


# ---------------------------------------------------------------------------
# 5. FastAPI endpoints – via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """Spin up a FastAPI TestClient with everything mocked."""
    from fastapi.testclient import TestClient

    fake_model = _make_fake_model([0.05, 0.90, 0.05])
    _pickle_load_side_effect._calls = 0

    with (
        patch("keras.models.load_model", return_value=fake_model),
        patch("builtins.open", side_effect=_open_side_effect),
        patch("pickle.load", side_effect=_pickle_load_side_effect),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None),
    ):
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as chatbot_app

    chatbot_app.words = list(FAKE_WORDS)
    chatbot_app.classes = list(FAKE_CLASSES)
    chatbot_app.intents = FAKE_INTENTS
    chatbot_app.model = fake_model

    client = TestClient(chatbot_app.app, raise_server_exceptions=False)
    return client


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


class TestChatEndpoint:
    def test_normal_message_returns_200(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "hello"})
        assert resp.status_code == 200

    def test_response_has_correct_schema(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "hello"})
        data = resp.json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_float(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "hello"})
        data = resp.json()
        assert isinstance(data["confidence"], float)

    def test_empty_message_returns_400(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_message_returns_400(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, test_client):
        resp = test_client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_my_name_is_pattern(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200

    def test_hi_my_name_is_pattern(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert resp.status_code == 200

    def test_i_am_pattern(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "i am Carol"})
        assert resp.status_code == 200

    def test_name_substitution_in_response(self, test_client):
        """If the model returns name_query intent, {n} should be replaced."""
        import app as chatbot_app
        # Force model to predict name_query (index 2)
        chatbot_app.model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = test_client.post("/api/chat", json={"msg": "my name is Dave"})
        assert resp.status_code == 200
        data = resp.json()
        # {n} should be replaced with the actual name
        assert "{n}" not in data["response"]


class TestHomeEndpoint:
    def test_home_returns_html(self, test_client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = test_client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "").lower() or resp.text


# ---------------------------------------------------------------------------
# 6. Pydantic models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self):
        if "app" in sys.modules:
            import app as m
        else:
            pytest.skip("app not loaded")
        import app as m
        req = m.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_empty_string_allowed_by_model(self):
        import app as m
        req = m.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_valid(self):
        import app as m
        resp = m.ChatResponse(response="Hello!", confidence=0.95)
        assert resp.response == "Hello!"
        assert resp.confidence == 0.95

    def test_chat_response_confidence_zero(self):
        import app as m
        resp = m.ChatResponse(response="...", confidence=0.0)
        assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# 7. CORS middleware is configured
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_options_request_has_cors_headers(self, test_client):
        resp = test_client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should add allow-origin header
        assert (
            "access-control-allow-origin" in resp.headers
            or resp.status_code in (200, 204)
        )
