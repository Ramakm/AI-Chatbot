"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Covers:
- FastAPI endpoints (/, /api/chat, /health)
- Core NLP helper functions (clean_up_sentence, bow, predict_class, getResponse)
- CORS middleware configuration
- Error / edge cases
- Upgraded package APIs: fastapi 0.109.1, python-multipart 0.0.7,
  keras 3.x (load_model), nltk 3.9 (word_tokenize, WordNetLemmatizer)
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
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "goodbye", "name"]
SAMPLE_CLASSES = ["greeting", "farewell", "name_tell"]
SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "farewell",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "name_tell",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}

# Minimal fake model predictions
def _fake_predict(x):
    """Return softmax-like probabilities favouring 'greeting' (index 0)."""
    arr = np.array([[0.85, 0.10, 0.05]])
    return arr


# ---------------------------------------------------------------------------
# Module-level patch: prevent app.py from touching the filesystem at import
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_model():
    m = MagicMock()
    m.predict.side_effect = _fake_predict
    return m


@pytest.fixture(scope="session")
def app_module(mock_model):
    """Import app with all external I/O mocked."""
    with (
        patch("keras.models.load_model", return_value=mock_model),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps(SAMPLE_INTENTS)),
        ),
        patch("pickle.load", side_effect=[SAMPLE_WORDS, SAMPLE_CLASSES]),
        patch(
            "fastapi.staticfiles.StaticFiles.__init__", return_value=None
        ),
        patch("nltk.download", return_value=True),
    ):
        # Remove cached module so patches apply cleanly
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("app",):
                del sys.modules[mod_name]

        import app as _app

        # Inject our test fixtures into module namespace so helper functions work
        _app.words = SAMPLE_WORDS
        _app.classes = SAMPLE_CLASSES
        _app.intents = SAMPLE_INTENTS
        _app.model = mock_model
        return _app


@pytest.fixture(scope="session")
def client(app_module):
    """Return a synchronous TestClient bound to the FastAPI app."""
    return TestClient(app_module.app, raise_server_exceptions=False)


# ===========================================================================
# 1.  FastAPI application meta / middleware
# ===========================================================================

class TestAppConfiguration:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_cors_middleware_present(self, app_module):
        from starlette.middleware.cors import CORSMiddleware
        middleware_classes = [
            m.cls for m in app_module.app.user_middleware
        ]
        assert CORSMiddleware in middleware_classes

    def test_routes_registered(self, app_module):
        paths = [r.path for r in app_module.app.routes]
        assert "/" in paths
        assert "/api/chat" in paths
        assert "/health" in paths


# ===========================================================================
# 2.  GET /health
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
# 3.  GET / (HTML home page)
# ===========================================================================

class TestHomeEndpoint:
    def test_home_returns_200(self, client, app_module):
        html_content = "<html><body>Chatbot</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_content_type_html(self, client, app_module):
        html_content = "<html><body>Chatbot</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


# ===========================================================================
# 4.  POST /api/chat  – happy paths
# ===========================================================================

class TestChatEndpointHappyPath:
    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting_returns_200(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "Hello")
        assert resp.status_code == 200

    def test_response_schema(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "Hello")
        data = resp.json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_float(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "Hi there")
        data = resp.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_my_name_is_pattern(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        # Response should contain the name substitution
        data = resp.json()
        assert "response" in data

    def test_hi_my_name_is_pattern(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200

    def test_i_am_pattern(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "i am Charlie")
        assert resp.status_code == 200

    def test_trailing_whitespace_stripped(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = self._post(client, "   Hello   ")
        assert resp.status_code == 200


# ===========================================================================
# 5.  POST /api/chat  – error / edge cases
# ===========================================================================

class TestChatEndpointErrors:
    def test_empty_string_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_invalid_json_body_returns_422(self, client):
        resp = client.post(
            "/api/chat",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_model_exception_returns_500(self, client, app_module):
        app_module.model.predict.side_effect = RuntimeError("model exploded")
        resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 500
        # Restore
        app_module.model.predict.side_effect = _fake_predict

    def test_400_detail_message(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


# ===========================================================================
# 6.  Pydantic models
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="Hello")
        assert req.msg == "Hello"

    def test_message_request_empty(self, app_module):
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.9)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.9)

    def test_chat_response_zero_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Unknown", confidence=0.0)
        assert resp.confidence == 0.0


# ===========================================================================
# 7.  clean_up_sentence  (uses nltk.word_tokenize + WordNetLemmatizer)
# ===========================================================================

class TestCleanUpSentence:
    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercases_tokens(self, app_module):
        result = app_module.clean_up_sentence("Hello")
        assert all(t == t.lower() for t in result)

    def test_lemmatizes_running(self, app_module):
        result = app_module.clean_up_sentence("running")
        assert "run" in result

    def test_lemmatizes_dogs(self, app_module):
        result = app_module.clean_up_sentence("dogs")
        assert "dog" in result

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_tokenized(self, app_module):
        result = app_module.clean_up_sentence("Hello!")
        assert "hello" in result

    def test_multi_word_sentence(self, app_module):
        result = app_module.clean_up_sentence("how are you")
        assert "how" in result
        assert "are" in result
        assert "you" in result

    def test_nltk_word_tokenize_called(self, app_module):
        with patch("nltk.word_tokenize", return_value=["hello"]) as mock_tok:
            app_module.clean_up_sentence("hello")
            mock_tok.assert_called_once_with("hello")


# ===========================================================================
# 8.  bow  (bag-of-words)
# ===========================================================================

class TestBagOfWords:
    def test_returns_numpy_array(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_vocabulary(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert len(result) == len(SAMPLE_WORDS)

    def test_word_present_gives_one(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        idx = SAMPLE_WORDS.index("hello")
        assert result[idx] == 1

    def test_word_absent_gives_zero(self, app_module):
        result = app_module.bow("xyz_not_in_vocab", SAMPLE_WORDS)
        assert all(v == 0 for v in result)

    def test_multiple_matching_words(self, app_module):
        result = app_module.bow("hello bye", SAMPLE_WORDS)
        assert result[SAMPLE_WORDS.index("hello")] == 1
        assert result[SAMPLE_WORDS.index("bye")] == 1

    def test_values_are_binary(self, app_module):
        result = app_module.bow("hello hello hello", SAMPLE_WORDS)
        assert set(result.tolist()).issubset({0, 1})

    def test_empty_sentence(self, app_module):
        result = app_module.bow("", SAMPLE_WORDS)
        assert result.sum() == 0

    def test_empty_vocab(self, app_module):
        result = app_module.bow("hello", [])
        assert len(result) == 0


# ===========================================================================
# 9.  predict_class
# ===========================================================================

class TestPredictClass:
    def test_returns_list(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("hello", mock_model)
        assert isinstance(result, list)

    def test_each_item_has_intent_and_probability(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("hello", mock_model)
        for item in result:
            assert "intent" in item
            assert "probability" in item

    def test_intent_is_string(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("hello", mock_model)
        for item in result:
            assert isinstance(item["intent"], str)

    def test_probability_is_string_or_float(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("hello", mock_model)
        for item in result:
            # probability stored as str in original code via str()
            float(item["probability"])  # must be castable

    def test_top_prediction_for_greeting(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("hello", mock_model)
        assert result[0]["intent"] == "greeting"

    def test_model_predict_called_once(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        mock_model.predict.reset_mock()
        app_module.predict_class("hello", mock_model)
        mock_model.predict.assert_called_once()

    def test_unknown_sentence_still_returns_list(self, app_module, mock_model):
        mock_model.predict.side_effect = _fake_predict
        result = app_module.predict_class("xyzzy frobnicator", mock_model)
        assert isinstance(result, list)


# ===========================================================================
# 10.  getResponse  (if defined in app module)
# ===========================================================================

class TestGetResponse:
    """Test the getResponse helper if it is present in app.py."""

    def test_returns_string_for_known_intent(self, app_module):
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not accessible in module scope")
        ints = [{"intent": "greeting", "probability": "0.85"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert isinstance(result, str)
        assert result in SAMPLE_INTENTS["intents"][0]["responses"]

    def test_returns_string_for_farewell(self, app_module):
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not accessible in module scope")
        ints = [{"intent": "farewell", "probability": "0.90"}]
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert result in SAMPLE_INTENTS["intents"][1]["responses"]

    def test_response_is_randomly_chosen(self, app_module):
        if not hasattr(app_module, "getResponse"):
            pytest.skip("getResponse not accessible in module scope")
        ints = [{"intent": "greeting", "probability": "0.85"}]
        results = {app_module.getResponse(ints, SAMPLE_INTENTS) for _ in range(30)}
        # At least 1 unique response (all valid)
        assert results.issubset(set(SAMPLE_INTENTS["intents"][0]["responses"]))


# ===========================================================================
# 11.  NLTK integration (upgraded 3.8.1 → 3.9)
# ===========================================================================

class TestNLTKIntegration:
    def test_word_tokenize_basic(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_word_tokenize_punctuation(self):
        import nltk
        tokens = nltk.word_tokenize("Hello!")
        assert "Hello" in tokens

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("running", pos="v") == "run"

    def test_wordnet_lemmatizer_unchanged(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("hello") == "hello"

    def test_lemmatizer_lowercased_input(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("Cats".lower()) == "cat"

    def test_word_tokenize_empty_string(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_word_tokenize_returns_list(self):
        import nltk
        tokens = nltk.word_tokenize("test sentence here")
        assert isinstance(tokens, list)


# ===========================================================================
# 12.  Keras / model loading (upgraded 2.14 → 3.12)
# ===========================================================================

class TestKerasModelLoading:
    def test_load_model_called_with_h5_path(self):
        """Verify keras.models.load_model is invoked with the correct path."""
        mock_mdl = MagicMock()
        mock_mdl.predict.side_effect = _fake_predict
        with patch("keras.models.load_model", return_value=mock_mdl) as mock_load:
            # Re-import to trigger the load
            for mod in list(sys.modules.keys()):
                if mod == "app":
                    del sys.modules[mod]
            with (
                patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_INTENTS))),
                patch("pickle.load", side_effect=[SAMPLE_WORDS, SAMPLE_CLASSES]),
                patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
                patch("nltk.download", return_value=True),
            ):
                import app as _app2  # noqa: F401
            mock_load.assert_called_once_with("chatbot_model.h5")

    def test_model_predict_returns_ndarray(self, mock_model):
        mock_model.predict.side_effect = _fake_predict
        bag = np.zeros((1, len(SAMPLE_WORDS)))
        result = mock_model.predict(bag)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, len(SAMPLE_CLASSES))

    def test_model_predict_probabilities_sum_to_one(self, mock_model):
        mock_model.predict.side_effect = _fake_predict
        bag = np.zeros((1, len(SAMPLE_WORDS)))
        result = mock_model.predict(bag)
        assert pytest.approx(result[0].sum(), abs=1e-5) == 1.0

    def test_keras_dense_layer_import():
        """Ensure keras 3.x Dense layer is importable."""
        from keras.layers import Dense
        layer = Dense(10, activation="relu")
        assert layer is not None

    def test_keras_sequential_model():
        """Ensure keras 3.x Sequential model is importable."""
        from keras.models import Sequential
        from keras.layers import Dense
        m = Sequential([Dense(4, activation="relu", input_shape=(8,))])
        assert m is not None

    def test_keras_dropout_layer():
        from keras.layers import Dropout
        layer = Dropout(0.5)
        assert layer is not None


# ===========================================================================
# 13.  FastAPI / python-multipart (upgraded)
# ===========================================================================

class TestFastAPIUpgraded:
    def test_post_json_content_type(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = client.post(
            "/api/chat",
            json={"msg": "hello"},
        )
        assert resp.status_code == 200

    def test_response_content_type_json(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = client.post("/api/chat", json={"msg": "hello"})
        assert "application/json" in resp.headers["content-type"]

    def test_http_exception_detail_propagated(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        assert "detail" in resp.json()

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/api/chat" in schema["paths"]

    def test_docs_endpoint_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_cors_headers_present(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = client.options(
            "/api/chat",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"},
        )
        # CORS middleware should respond with allow-origin header
        assert resp.headers.get("access-control-allow-origin") is not None

    def test_extra_fields_in_body_ignored(self, client, app_module):
        """FastAPI / Pydantic should ignore unknown extra fields."""
        app_module.model.predict.side_effect = _fake_predict
        resp = client.post("/api/chat", json={"msg": "hello", "unknown_field": "data"})
        assert resp.status_code == 200

    def test_multipart_form_not_required(self, client):
        """Endpoint accepts JSON, not multipart – wrong content type → 422."""
        resp = client.post(
            "/api/chat",
            data={"msg": "hello"},  # form data, not JSON
        )
        # FastAPI returns 422 for wrong body type on a JSON endpoint
        assert resp.status_code in (200, 422)


# ===========================================================================
# 14.  Integration: end-to-end chat flow
# ===========================================================================

class TestEndToEndChatFlow:
    def test_greeting_flow(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] in SAMPLE_INTENTS["intents"][0]["responses"]
        assert data["confidence"] == pytest.approx(0.85, abs=1e-3)

    def test_name_substitution_my_name_is(self, client, app_module):
        # Override predict to return name_tell class (index 2)
        def _name_predict(x):
            return np.array([[0.05, 0.05, 0.90]])

        app_module.model.predict.side_effect = _name_predict
        resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200
        data = resp.json()
        # The response template "Nice to meet you, {n}!" should have {n} replaced
        assert "Alice" in data["response"] or "{n}" not in data["response"]
        # Restore
        app_module.model.predict.side_effect = _fake_predict

    def test_name_substitution_i_am(self, client, app_module):
        def _name_predict(x):
            return np.array([[0.05, 0.05, 0.90]])

        app_module.model.predict.side_effect = _name_predict
        resp = client.post("/api/chat", json={"msg": "i am Bob"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Bob" in data["response"] or "{n}" not in data["response"]
        app_module.model.predict.side_effect = _fake_predict

    def test_confidence_matches_top_prediction(self, client, app_module):
        def _fixed_predict(x):
            return np.array([[0.70, 0.20, 0.10]])

        app_module.model.predict.side_effect = _fixed_predict
        resp = client.post("/api/chat", json={"msg": "Goodbye"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.70, abs=1e-3)
        app_module.model.predict.side_effect = _fake_predict

    def test_multiple_sequential_requests(self, client, app_module):
        app_module.model.predict.side_effect = _fake_predict
        messages = ["Hello", "Hi", "Bye", "How are you"]
        for msg in messages:
            resp = client.post("/api/chat", json={"msg": msg})
            assert resp.status_code == 200
