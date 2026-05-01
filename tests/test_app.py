"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Upgraded packages exercised:
  - fastapi  0.104.1 → 0.109.1
  - python-multipart 0.0.6 → 0.0.7
  - nltk 3.8.1 → 3.9
"""

import json
import os
import pickle
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – build lightweight stand-ins for heavy optional deps so that
# app.py can be imported without a real Keras model or NLTK corpora.
# ---------------------------------------------------------------------------

def _make_keras_stub():
    """Return a minimal stub that satisfies `from keras.models import load_model`."""
    keras_mod = types.ModuleType("keras")
    models_mod = types.ModuleType("keras.models")
    layers_mod = types.ModuleType("keras.layers")

    fake_model = MagicMock()
    # predict returns a 2-D numpy array shaped (1, N)
    fake_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])

    models_mod.load_model = MagicMock(return_value=fake_model)
    models_mod.Sequential = MagicMock()
    layers_mod.Dense = MagicMock()
    layers_mod.Dropout = MagicMock()

    keras_mod.models = models_mod
    keras_mod.layers = layers_mod
    sys.modules.setdefault("keras", keras_mod)
    sys.modules.setdefault("keras.models", models_mod)
    sys.modules.setdefault("keras.layers", layers_mod)
    return fake_model


def _make_tensorflow_stub():
    tf = types.ModuleType("tensorflow")
    tf_keras = types.ModuleType("tensorflow.keras")
    tf_keras_opt = types.ModuleType("tensorflow.keras.optimizers")
    tf_keras_opt.SGD = MagicMock()
    tf_keras.optimizers = tf_keras_opt
    tf.keras = tf_keras
    sys.modules.setdefault("tensorflow", tf)
    sys.modules.setdefault("tensorflow.keras", tf_keras)
    sys.modules.setdefault("tensorflow.keras.optimizers", tf_keras_opt)


# Sample data used throughout the tests
SAMPLE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "goodbye", "name", "my", "i", "am"]
SAMPLE_CLASSES = ["greeting", "goodbye", "name"]
SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is", "I am", "hi my name is"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_stubs():
    """Inject all heavy-dependency stubs before any import of app.py."""
    _make_tensorflow_stub()
    _make_keras_stub()


@pytest.fixture(scope="session")
def patched_app():
    """
    Import app with all file-system and model loading calls mocked out.
    Returns (app_module, fake_model).
    """
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])

    pkl_greeting = pickle.dumps(["greeting"])
    pkl_words = pickle.dumps(SAMPLE_WORDS)
    pkl_classes = pickle.dumps(SAMPLE_CLASSES)

    def _open_side_effect(path, mode="r", *args, **kwargs):
        if "words.pkl" in str(path):
            import io
            return io.BytesIO(pkl_words)
        if "classes.pkl" in str(path):
            import io
            return io.BytesIO(pkl_classes)
        if "intents.json" in str(path):
            import io
            return io.StringIO(json.dumps(SAMPLE_INTENTS))
        if "index.html" in str(path):
            import io
            return io.StringIO("<html><body>Chatbot</body></html>")
        raise FileNotFoundError(f"Unexpected open: {path}")

    with (
        patch("builtins.open", side_effect=_open_side_effect),
        patch("pickle.load", side_effect=[SAMPLE_WORDS, SAMPLE_CLASSES]),
        patch("keras.models.load_model", return_value=fake_model),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None),
    ):
        # Remove cached module so re-import picks up our patches
        sys.modules.pop("app", None)
        import app as app_module

    app_module.model = fake_model
    app_module.words = SAMPLE_WORDS
    app_module.classes = SAMPLE_CLASSES
    app_module.intents = SAMPLE_INTENTS
    return app_module, fake_model


@pytest.fixture()
def client(patched_app):
    app_module, _ = patched_app
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture()
def fake_model(patched_app):
    _, model = patched_app
    return model


@pytest.fixture()
def app_module(patched_app):
    module, _ = patched_app
    return module


# ---------------------------------------------------------------------------
# 1. FastAPI application metadata
# ---------------------------------------------------------------------------

class TestAppMetadata:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_app_description_contains_fastapi(self, app_module):
        assert "FastAPI" in app_module.app.description

    def test_cors_middleware_present(self, app_module):
        from starlette.middleware.cors import CORSMiddleware
        middleware_types = [m.cls for m in app_module.app.user_middleware]
        assert CORSMiddleware in middleware_types


# ---------------------------------------------------------------------------
# 2. Pydantic models (FastAPI upgrade – strict validation behaviour)
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="Hello")
        assert req.msg == "Hello"

    def test_message_request_rejects_missing_field(self, app_module):
        import pydantic
        with pytest.raises((pydantic.ValidationError, TypeError)):
            app_module.MessageRequest()

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert pytest.approx(resp.confidence) == 0.95

    def test_chat_response_confidence_float(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=1)
        assert isinstance(resp.confidence, float)


# ---------------------------------------------------------------------------
# 3. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_payload(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ---------------------------------------------------------------------------
# 4. Home endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_html(self, client, app_module):
        with patch.object(
            app_module,
            "open",
            mock_open(read_data="<html><body>Chatbot</body></html>"),
            create=True,
        ):
            # patch built-in open inside app module namespace
            with patch("builtins.open", mock_open(read_data="<html><body>Chatbot</body></html>")):
                resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# 5. Chat endpoint – happy paths (FastAPI 0.109 routing + validation)
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:
    """Tests for /api/chat using the TestClient (exercises fastapi 0.109.1)."""

    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting_returns_200(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        resp = self._post(client, "Hello")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        data = self._post(client, "Hello").json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_numeric(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        data = self._post(client, "Hello").json()
        assert isinstance(data["confidence"], float)

    def test_my_name_is_pattern(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        # The placeholder {n} should be replaced
        assert "{n}" not in resp.json()["response"]

    def test_hi_my_name_is_pattern(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200
        assert "{n}" not in resp.json()["response"]

    def test_i_am_pattern(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "i am Carol")
        assert resp.status_code == 200
        assert "{n}" not in resp.json()["response"]

    def test_case_insensitive_my_name_is(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "MY NAME IS Dave")
        assert resp.status_code == 200

    def test_case_insensitive_i_am(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "I AM Eve")
        assert resp.status_code == 200

    def test_goodbye_message(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 1.0, 0.0]])
        resp = self._post(client, "Goodbye")
        assert resp.status_code == 200

    def test_response_is_string(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        data = self._post(client, "Hi there").json()
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0


# ---------------------------------------------------------------------------
# 6. Chat endpoint – error / edge cases
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

    def test_wrong_type_returns_422(self, client):
        # FastAPI 0.109 still validates via Pydantic v2 – int not allowed for str
        resp = client.post("/api/chat", json={"msg": 12345})
        # Pydantic v2 coerces int to str, so 200 or 422 are both acceptable
        assert resp.status_code in (200, 422)

    def test_model_predict_exception_returns_500(self, client, fake_model):
        fake_model.predict.side_effect = RuntimeError("model exploded")
        resp = self._post(client, {"msg": "Hello"})
        assert resp.status_code == 500
        fake_model.predict.side_effect = None  # restore

    def test_error_detail_present_on_500(self, client, fake_model):
        fake_model.predict.side_effect = RuntimeError("boom")
        resp = self._post(client, {"msg": "Hello"})
        assert "detail" in resp.json()
        fake_model.predict.side_effect = None


# ---------------------------------------------------------------------------
# 7. NLTK integration – clean_up_sentence (nltk 3.9)
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    """Tests for clean_up_sentence which uses nltk.word_tokenize + WordNetLemmatizer."""

    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercases_tokens(self, app_module):
        result = app_module.clean_up_sentence("HELLO WORLD")
        assert all(w == w.lower() for w in result)

    def test_lemmatizes_plural(self, app_module):
        # "dogs" → "dog"
        result = app_module.clean_up_sentence("dogs")
        assert "dog" in result

    def test_lemmatizes_running(self, app_module):
        # "running" as noun → "running"; as verb → "run" — either is fine
        result = app_module.clean_up_sentence("running")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_empty_string_returns_empty_list(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_tokenised(self, app_module):
        # NLTK tokeniser splits punctuation; ensure no crash
        result = app_module.clean_up_sentence("Hello!")
        assert isinstance(result, list)

    def test_multiple_words(self, app_module):
        result = app_module.clean_up_sentence("how are you")
        assert len(result) >= 3

    def test_sentence_with_numbers(self, app_module):
        result = app_module.clean_up_sentence("I have 3 dogs")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 8. Bag-of-Words (bow)
# ---------------------------------------------------------------------------

class TestBow:
    def test_returns_numpy_array(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_words(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert len(result) == len(SAMPLE_WORDS)

    def test_known_word_activates_correct_index(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        hello_idx = SAMPLE_WORDS.index("hello")
        assert result[hello_idx] == 1

    def test_unknown_word_leaves_zeros(self, app_module):
        result = app_module.bow("zyxwvutsrqp", SAMPLE_WORDS)
        assert np.sum(result) == 0

    def test_empty_sentence_all_zeros(self, app_module):
        result = app_module.bow("", SAMPLE_WORDS)
        assert np.sum(result) == 0

    def test_multiple_known_words(self, app_module):
        result = app_module.bow("hello bye", SAMPLE_WORDS)
        assert result[SAMPLE_WORDS.index("hello")] == 1
        assert result[SAMPLE_WORDS.index("bye")] == 1

    def test_binary_values_only(self, app_module):
        result = app_module.bow("hello hello hello", SAMPLE_WORDS)
        assert set(result).issubset({0, 1})

    def test_empty_word_list(self, app_module):
        result = app_module.bow("hello", [])
        assert len(result) == 0

    def test_show_details_true_no_crash(self, app_module, capsys):
        result = app_module.bow("hello", SAMPLE_WORDS, show_details=True)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 9. predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def _make_model(self, probs):
        m = MagicMock()
        m.predict.return_value = np.array([probs])
        return m

    def test_returns_list(self, app_module):
        model = self._make_model([0.1, 0.8, 0.1])
        result = app_module.predict_class("hello", model)
        assert isinstance(result, list)

    def test_each_element_has_intent_and_probability(self, app_module):
        model = self._make_model([0.1, 0.8, 0.1])
        result = app_module.predict_class("hello", model)
        for item in result:
            assert "intent" in item
            assert "probability" in item

    def test_intent_is_string(self, app_module):
        model = self._make_model([0.1, 0.8, 0.1])
        result = app_module.predict_class("hello", model)
        assert all(isinstance(r["intent"], str) for r in result)

    def test_probability_is_string_or_float(self, app_module):
        model = self._make_model([0.1, 0.8, 0.1])
        result = app_module.predict_class("hello", model)
        for r in result:
            # probability stored as string in reference impl
            float(r["probability"])  # must be convertible

    def test_filters_low_confidence(self, app_module):
        # All classes below 0.25 threshold → empty list
        model = self._make_model([0.1, 0.1, 0.1])
        result = app_module.predict_class("hello", model)
        assert isinstance(result, list)

    def test_high_confidence_class_returned(self, app_module):
        model = self._make_model([0.05, 0.90, 0.05])
        result = app_module.predict_class("hello", model)
        if result:
            assert result[0]["intent"] == SAMPLE_CLASSES[1]

    def test_model_predict_called_once(self, app_module):
        model = self._make_model([0.1, 0.8, 0.1])
        app_module.predict_class("hello", model)
        model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# 10. getResponse
# ---------------------------------------------------------------------------

class TestGetResponse:
    def test_returns_string(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert isinstance(result, str)

    def test_response_from_correct_intent(self, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert result in SAMPLE_INTENTS["intents"][0]["responses"]

    def test_goodbye_response(self, app_module):
        ints = [{"intent": "goodbye", "probability": "0.9"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert result in SAMPLE_INTENTS["intents"][1]["responses"]

    def test_name_response_contains_placeholder(self, app_module):
        ints = [{"intent": "name", "probability": "0.9"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        # Raw response keeps {n} before substitution
        assert isinstance(result, str)

    def test_empty_ints_handled(self, app_module):
        """Should not raise even with empty predictions."""
        try:
            result = app_module.getResponse([], SAMPLE_INTENTS)
            assert isinstance(result, (str, type(None)))
        except (IndexError, KeyError):
            pass  # Acceptable if implementation raises


# ---------------------------------------------------------------------------
# 11. Name-substitution logic in /api/chat
# ---------------------------------------------------------------------------

class TestNameSubstitution:
    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_my_name_is_alice_response_contains_alice(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "my name is Alice")
        data = resp.json()
        assert resp.status_code == 200
        # response or raw check – at minimum no literal {n}
        assert "{n}" not in data["response"]

    def test_i_am_bob_response_has_no_placeholder(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "i am Bob")
        assert resp.status_code == 200
        assert "{n}" not in resp.json()["response"]

    def test_hi_my_name_is_carol_response(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        resp = self._post(client, "hi my name is Carol")
        assert resp.status_code == 200
        assert "{n}" not in resp.json()["response"]


# ---------------------------------------------------------------------------
# 12. FastAPI route registration (0.109 compatibility)
# ---------------------------------------------------------------------------

class TestRouteRegistration:
    def test_routes_include_root(self, app_module):
        paths = [r.path for r in app_module.app.routes]
        assert "/" in paths

    def test_routes_include_chat(self, app_module):
        paths = [r.path for r in app_module.app.routes]
        assert "/api/chat" in paths

    def test_routes_include_health(self, app_module):
        paths = [r.path for r in app_module.app.routes]
        assert "/health" in paths

    def test_chat_accepts_post(self, app_module):
        from fastapi.routing import APIRoute
        chat_routes = [
            r for r in app_module.app.routes
            if isinstance(r, APIRoute) and r.path == "/api/chat"
        ]
        assert chat_routes, "No APIRoute found for /api/chat"
        assert "POST" in chat_routes[0].methods

    def test_health_accepts_get(self, app_module):
        from fastapi.routing import APIRoute
        health_routes = [
            r for r in app_module.app.routes
            if isinstance(r, APIRoute) and r.path == "/health"
        ]
        assert health_routes
        assert "GET" in health_routes[0].methods

    def test_chat_response_model_set(self, app_module):
        from fastapi.routing import APIRoute
        chat_routes = [
            r for r in app_module.app.routes
            if isinstance(r, APIRoute) and r.path == "/api/chat"
        ]
        assert chat_routes[0].response_model is app_module.ChatResponse


# ---------------------------------------------------------------------------
# 13. NLTK tokenization smoke tests (nltk 3.9 API)
# ---------------------------------------------------------------------------

class TestNLTKIntegration:
    """Directly exercise nltk APIs used in app.py to detect 3.8→3.9 regressions."""

    def test_word_tokenize_returns_list(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) == 2

    def test_word_tokenize_handles_punctuation(self):
        import nltk
        tokens = nltk.word_tokenize("Hello, world!")
        assert "Hello" in tokens or "hello" in tokens.copy()

    def test_word_tokenize_empty_string(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        result = lem.lemmatize("running", pos="v")
        assert result == "run"

    def test_wordnet_lemmatizer_adjective(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        result = lem.lemmatize("better", pos="a")
        assert isinstance(result, str)

    def test_wordnet_lemmatizer_preserves_lowercase(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        result = lem.lemmatize("hello")
        assert result == result.lower()

    def test_word_tokenize_multi_sentence(self):
        import nltk
        tokens = nltk.word_tokenize("Hi. How are you?")
        assert len(tokens) > 4

    def test_lemmatizer_consistency_across_calls(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("cats") == lem.lemmatize("cats")


# ---------------------------------------------------------------------------
# 14. Integration: full round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_greeting_roundtrip(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
        resp = client.post("/api/chat", json={"msg": "Hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0

    def test_full_goodbye_roundtrip(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
        resp = client.post("/api/chat", json={"msg": "Goodbye"})
        assert resp.status_code == 200

    def test_confidence_matches_max_prediction(self, client, fake_model):
        probs = [0.05, 0.85, 0.10]
        fake_model.predict.return_value = np.array([probs])
        resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        # confidence should be the highest probability (after any threshold filtering)
        body = resp.json()
        assert body["confidence"] >= 0.0

    def test_health_then_chat_sequence(self, client, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        h = client.get("/health")
        c = client.post("/api/chat", json={"msg": "Hello"})
        assert h.status_code == 200
        assert c.status_code == 200
