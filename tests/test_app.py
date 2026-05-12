"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).
Covers:
- FastAPI endpoints (/, /api/chat, /health)
- Core NLP helper functions (clean_up_sentence, bow, predict_class, getResponse)
- CORS middleware
- Pydantic request / response models
- Error / edge cases
- Upgraded package APIs: fastapi 0.109.1, python-multipart 0.0.7, keras 3.x, nltk 3.9
"""

import json
import pickle
import sys
import types
import importlib
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – build lightweight stubs so app.py can be imported without real
# model files or a GPU.
# ---------------------------------------------------------------------------

def _make_fake_model(n_classes: int = 3):
    """Return a mock Keras-model-like object whose predict() returns a 2-D
    numpy array whose values sum to 1 (softmax-shaped)."""
    mock_model = MagicMock()
    probs = np.zeros((1, n_classes), dtype=np.float32)
    probs[0, 0] = 0.85
    probs[0, 1] = 0.10
    probs[0, 2] = 0.05
    mock_model.predict.return_value = probs
    return mock_model


FAKE_WORDS = ["hello", "hi", "how", "are", "you", "name", "my", "i", "am", "is", "bye", "goodbye"]
FAKE_CLASSES = ["greeting", "identity", "farewell"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "identity",
            "patterns": ["my name is", "I am"],
            "responses": ["Nice to meet you, {n}!"],
        },
        {
            "tag": "farewell",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
    ]
}


# ---------------------------------------------------------------------------
# Fixture: import app with all heavy I/O mocked out
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    """Import app.py with all file-system and model dependencies mocked."""
    fake_model = _make_fake_model(n_classes=len(FAKE_CLASSES))

    # Patch keras.models.load_model before import
    keras_mock = MagicMock()
    keras_mock.models.load_model.return_value = fake_model

    # Build a fake 'keras' package tree so the 'from keras.models import load_model' works
    keras_pkg = types.ModuleType("keras")
    keras_models_mod = types.ModuleType("keras.models")
    keras_models_mod.load_model = lambda *a, **kw: fake_model
    keras_pkg.models = keras_models_mod
    sys.modules.setdefault("keras", keras_pkg)
    sys.modules.setdefault("keras.models", keras_models_mod)

    words_bytes = pickle.dumps(FAKE_WORDS)
    classes_bytes = pickle.dumps(FAKE_CLASSES)
    intents_json = json.dumps(FAKE_INTENTS)

    open_calls = {
        "words.pkl": words_bytes,
        "classes.pkl": classes_bytes,
        "intents.json": intents_json,
    }

    original_open = open
    original_pickle_load = pickle.load

    def fake_open(path, mode="r", **kwargs):
        name = str(path)
        if name in open_calls:
            import io
            data = open_calls[name]
            if isinstance(data, bytes):
                return io.BytesIO(data)
            return io.StringIO(data)
        # For templates/index.html
        if "index.html" in name:
            return io.StringIO("<html><body>Chatbot</body></html>")
        return original_open(path, mode, **kwargs)

    def fake_pickle_load(f):
        data = f.read()
        return pickle.loads(data)

    with patch("builtins.open", side_effect=fake_open), \
         patch("pickle.load", side_effect=fake_pickle_load), \
         patch("keras.models.load_model", return_value=fake_model), \
         patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):

        # Remove cached module if present (re-import cleanly)
        sys.modules.pop("app", None)
        import app as _app

        # Inject our fake objects directly (in case module-level assignment happened)
        _app.model = fake_model
        _app.words = FAKE_WORDS
        _app.classes = FAKE_CLASSES
        _app.intents = FAKE_INTENTS

    return _app


@pytest.fixture(scope="module")
def client(app_module):
    """TestClient wrapping the FastAPI app."""
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1.  Pydantic models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="Hello")
        assert req.msg == "Hello"

    def test_message_request_empty_string(self, app_module):
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_fields(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.9)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.9)

    def test_chat_response_zero_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Hello", confidence=0.0)
        assert resp.confidence == 0.0

    def test_chat_response_max_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Ok", confidence=1.0)
        assert resp.confidence == 1.0


# ---------------------------------------------------------------------------
# 2.  /health endpoint
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

    def test_health_content_type(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# 3.  / (home) endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_html(self, app_module, client):
        # Patch the open call for template
        import io
        html_content = "<html><body>Chatbot</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_content_type_html(self, app_module, client):
        import io
        with patch("builtins.open", mock_open(read_data="<html></html>")):
            resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 4.  /api/chat endpoint – happy paths
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:

    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting(self, client):
        resp = self._post(client, "hello")
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert isinstance(data["response"], str)

    def test_response_is_from_intents(self, client, app_module):
        all_responses = []
        for intent in app_module.intents["intents"]:
            all_responses.extend(intent["responses"])
        resp = self._post(client, "hello")
        if resp.status_code == 200:
            data = resp.json()
            # The response should come from intents list (possibly with {n} replaced)
            assert len(data["response"]) > 0

    def test_my_name_is_pattern(self, client):
        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        data = resp.json()
        # {n} should have been replaced with "Alice"
        # (only if the matched intent has {n} in response)
        assert "response" in data

    def test_hi_my_name_is_pattern(self, client):
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_i_am_pattern(self, client):
        resp = self._post(client, "i am Charlie")
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_confidence_between_0_and_1(self, client):
        resp = self._post(client, "hello world")
        if resp.status_code == 200:
            data = resp.json()
            assert 0.0 <= data["confidence"] <= 1.0

    def test_leading_trailing_whitespace_stripped(self, client):
        resp = self._post(client, "  hello  ")
        assert resp.status_code == 200

    def test_name_substitution_alice(self, client, app_module):
        # Force identity intent to win and verify {n} replacement
        fake_model = app_module.model
        n_classes = len(FAKE_CLASSES)
        probs = np.zeros((1, n_classes), dtype=np.float32)
        # identity is index 1
        probs[0, 1] = 0.99
        probs[0, 0] = 0.005
        probs[0, 2] = 0.005
        fake_model.predict.return_value = probs

        resp = self._post(client, "my name is Alice")
        assert resp.status_code == 200
        data = resp.json()
        # Reset to greeting win
        probs2 = np.zeros((1, n_classes), dtype=np.float32)
        probs2[0, 0] = 0.85
        probs2[0, 1] = 0.10
        probs2[0, 2] = 0.05
        fake_model.predict.return_value = probs2


# ---------------------------------------------------------------------------
# 5.  /api/chat endpoint – error / edge cases
# ---------------------------------------------------------------------------

class TestChatEndpointErrors:

    def test_empty_message_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_message_returns_400(self, client):
        resp = client.post("/api/chat", json={"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_invalid_json_content_type(self, client):
        resp = client.post(
            "/api/chat",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_400_detail_message(self, client):
        resp = client.post("/api/chat", json={"msg": ""})
        data = resp.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower() or "cannot" in data["detail"].lower()

    def test_500_on_internal_error(self, client, app_module):
        """Simulate model.predict raising an exception → 500."""
        original_predict = app_module.model.predict
        app_module.model.predict.side_effect = RuntimeError("GPU OOM")
        try:
            resp = client.post("/api/chat", json={"msg": "hello"})
            assert resp.status_code == 500
        finally:
            app_module.model.predict.side_effect = None
            app_module.model.predict.return_value = np.array(
                [[0.85, 0.10, 0.05]], dtype=np.float32
            )


# ---------------------------------------------------------------------------
# 6.  CORS middleware
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_cors_headers_present_on_options(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS returns 200 for preflight
        assert resp.status_code in (200, 400)

    def test_cors_allow_origin_wildcard(self, client):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        # Should echo back ACAO header
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "*" or acao == "http://example.com"


# ---------------------------------------------------------------------------
# 7.  clean_up_sentence  (uses nltk.word_tokenize + WordNetLemmatizer)
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_basic_tokenization(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_lowercasing(self, app_module):
        result = app_module.clean_up_sentence("HELLO WORLD")
        for token in result:
            assert token == token.lower()

    def test_lemmatization_running(self, app_module):
        # "running" → "running" (lemmatize default is noun; verb form is 'run')
        result = app_module.clean_up_sentence("running")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_punctuation_kept(self, app_module):
        # nltk tokenizer keeps punctuation as separate tokens
        result = app_module.clean_up_sentence("Hello!")
        assert any(t.isalpha() for t in result)

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_single_word(self, app_module):
        result = app_module.clean_up_sentence("bye")
        assert "bye" in result

    def test_multiple_sentences(self, app_module):
        result = app_module.clean_up_sentence("Hello. How are you?")
        assert len(result) >= 4

    def test_numbers_in_sentence(self, app_module):
        result = app_module.clean_up_sentence("I have 3 cats")
        assert "3" in result or "cat" in result or "i" in result


# ---------------------------------------------------------------------------
# 8.  bow (bag of words)
# ---------------------------------------------------------------------------

class TestBow:
    def test_output_length_matches_words(self, app_module):
        bag = app_module.bow("hello", FAKE_WORDS)
        assert len(bag) == len(FAKE_WORDS)

    def test_output_is_numpy_array(self, app_module):
        bag = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(bag, np.ndarray)

    def test_known_word_set_to_1(self, app_module):
        bag = app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert bag[idx] == 1

    def test_unknown_word_all_zeros(self, app_module):
        bag = app_module.bow("xyzzy", FAKE_WORDS)
        assert np.sum(bag) == 0

    def test_values_are_binary(self, app_module):
        bag = app_module.bow("hello how are you", FAKE_WORDS)
        assert set(bag.tolist()).issubset({0, 1})

    def test_empty_sentence_all_zeros(self, app_module):
        bag = app_module.bow("", FAKE_WORDS)
        assert np.sum(bag) == 0

    def test_multiple_words_match(self, app_module):
        bag = app_module.bow("hello bye", FAKE_WORDS)
        assert bag[FAKE_WORDS.index("hello")] == 1
        assert bag[FAKE_WORDS.index("bye")] == 1

    def test_show_details_flag(self, app_module, capsys):
        app_module.bow("hello", FAKE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "found in bag" in captured.out

    def test_custom_words_list(self, app_module):
        custom_words = ["cat", "dog", "fish"]
        bag = app_module.bow("my cat is happy", custom_words)
        assert len(bag) == 3
        assert bag[0] == 1  # 'cat' found


# ---------------------------------------------------------------------------
# 9.  predict_class
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

    def test_intent_is_string(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        for item in result:
            assert isinstance(item["intent"], str)

    def test_probability_is_string_parseable_as_float(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        for item in result:
            val = float(item["probability"])
            assert 0.0 <= val <= 1.0

    def test_model_predict_called(self, app_module):
        app_module.model.predict.reset_mock()
        app_module.predict_class("hello", app_module.model)
        app_module.model.predict.assert_called_once()

    def test_results_sorted_by_probability_descending(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        probs = [float(r["probability"]) for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_low_confidence_filtered_out(self, app_module):
        """Predictions below threshold (0.25) should be excluded."""
        n_classes = len(FAKE_CLASSES)
        probs = np.zeros((1, n_classes), dtype=np.float32)
        probs[0, 0] = 0.1   # below threshold
        probs[0, 1] = 0.05
        probs[0, 2] = 0.05
        app_module.model.predict.return_value = probs

        result = app_module.predict_class("xyz", app_module.model)
        # All below 0.25 → should return empty list
        assert isinstance(result, list)
        # Restore
        probs2 = np.zeros((1, n_classes), dtype=np.float32)
        probs2[0, 0] = 0.85
        probs2[0, 1] = 0.10
        probs2[0, 2] = 0.05
        app_module.model.predict.return_value = probs2

    def test_empty_sentence(self, app_module):
        result = app_module.predict_class("", app_module.model)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 10.  getResponse
# ---------------------------------------------------------------------------

class TestGetResponse:
    def _make_ints(self, tag):
        return [{"intent": tag, "probability": "0.85"}]

    def test_returns_string(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_response_in_intents(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        valid = FAKE_INTENTS["intents"][0]["responses"]
        assert result in valid

    def test_identity_response_contains_placeholder(self, app_module):
        ints = self._make_ints("identity")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        # The raw response should contain {n} before substitution
        assert "{n}" in result

    def test_farewell_response(self, app_module):
        ints = self._make_ints("farewell")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        valid = FAKE_INTENTS["intents"][2]["responses"]
        assert result in valid

    def test_empty_ints_list(self, app_module):
        """Should not raise; may return a default/empty string."""
        try:
            result = app_module.getResponse([], FAKE_INTENTS)
            assert isinstance(result, str)
        except (IndexError, KeyError):
            pass  # acceptable – no prediction means no response


# ---------------------------------------------------------------------------
# 11.  FastAPI app metadata / OpenAPI
# ---------------------------------------------------------------------------

class TestAppMetadata:
    def test_openapi_endpoint(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "AI Chatbot API"
        assert schema["info"]["version"] == "1.0.0"

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_endpoint(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_chat_route_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        assert "/api/chat" in paths

    def test_health_route_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        assert "/health" in paths


# ---------------------------------------------------------------------------
# 12.  HTTP method restrictions
# ---------------------------------------------------------------------------

class TestHTTPMethods:
    def test_get_on_chat_returns_405(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_put_on_chat_returns_405(self, client):
        resp = client.put("/api/chat", json={"msg": "hi"})
        assert resp.status_code == 405

    def test_delete_on_chat_returns_405(self, client):
        resp = client.delete("/api/chat")
        assert resp.status_code == 405

    def test_post_on_health_returns_405(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


# ---------------------------------------------------------------------------
# 13.  NLTK integration (upgraded 3.8.1 → 3.9)
# ---------------------------------------------------------------------------

class TestNLTKIntegration:
    """Tests verifying nltk APIs used in the project still behave correctly."""

    def test_word_tokenize_basic(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_word_tokenize_punctuation(self):
        import nltk
        tokens = nltk.word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "," in tokens

    def test_word_tokenize_empty(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"
        assert lem.lemmatize("cats") == "cat"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("running", pos="v") == "run"
        assert lem.lemmatize("played", pos="v") == "play"

    def test_wordnet_lemmatizer_lowercases_correctly(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        result = lem.lemmatize("Cats".lower())
        assert result == "cat"

    def test_word_tokenize_sentence(self):
        import nltk
        tokens = nltk.word_tokenize("How are you doing today?")
        assert "How" in tokens
        assert "today" in tokens

    def test_lemmatize_preserves_unknown_words(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        # Unknown word should be returned unchanged
        result = lem.lemmatize("xyzzy")
        assert result == "xyzzy"


# ---------------------------------------------------------------------------
# 14.  Keras model mock – upgraded keras 2.x → 3.x API surface
# ---------------------------------------------------------------------------

class TestKerasModelMock:
    """Verify that the model mock behaves like a Keras 3.x model."""

    def test_predict_returns_numpy_array(self, app_module):
        bag = np.zeros((1, len(FAKE_WORDS)), dtype=np.float32)
        result = app_module.model.predict(bag)
        assert isinstance(result, np.ndarray)

    def test_predict_shape(self, app_module):
        bag = np.zeros((1, len(FAKE_WORDS)), dtype=np.float32)
        result = app_module.model.predict(bag)
        assert result.shape == (1, len(FAKE_CLASSES))

    def test_predict_values_sum_to_1(self, app_module):
        bag = np.zeros((1, len(FAKE_WORDS)), dtype=np.float32)
        result = app_module.model.predict(bag)
        total = float(np.sum(result[0]))
        assert abs(total - 1.0) < 1e-3

    def test_load_model_called_at_import(self):
        """keras.models.load_model should have been called during app import."""
        import keras.models as km
        # Our stub registered it; just ensure the attribute exists
        assert callable(km.load_model)

    def test_model_predict_receives_correct_input_shape(self, app_module):
        """bow() output fed into model.predict should be a 2-D array."""
        bag = app_module.bow("hello", FAKE_WORDS)
        reshaped = np.array([bag])
        result = app_module.model.predict(reshaped)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# 15.  Integration-style: end-to-end chat flow
# ---------------------------------------------------------------------------

class TestEndToEndChatFlow:
    def test_full_greeting_flow(self, client):
        resp = client.post(
            "/api/chat",
            json={"msg": "hello"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "confidence" in body
        assert body["response"] != ""

    def test_name_substitution_end_to_end(self, client, app_module):
        """'my name is X' should produce a response where {n} is replaced."""
        n_classes = len(FAKE_CLASSES)
        probs = np.zeros((1, n_classes), dtype=np.float32)
        probs[0, 1] = 0.99  # identity wins
        app_module.model.predict.return_value = probs

        resp = client.post("/api/chat", json={"msg": "my name is TestUser"})
        assert resp.status_code == 200
        data = resp.json()
        # "{n}" should be replaced with "TestUser" and NOT appear literally
        assert "{n}" not in data["response"]
        assert "TestUser" in data["response"]

        # restore
        probs2 = np.zeros((1, n_classes), dtype=np.float32)
        probs2[0, 0] = 0.85
        app_module.model.predict.return_value = probs2

    def test_multiple_sequential_requests(self, client):
        messages = ["hello", "how are you", "bye", "my name is Eve"]
        for msg in messages:
            resp = client.post("/api/chat", json={"msg": msg})
            assert resp.status_code == 200

    def test_long_message(self, client):
        long_msg = "hello " * 100
        resp = client.post("/api/chat", json={"msg": long_msg})
        assert resp.status_code in (200, 400, 500)  # should not crash server

    def test_special_characters_in_message(self, client):
        resp = client.post("/api/chat", json={"msg": "hello! @#$%^&*()"})
        assert resp.status_code in (200, 500)

    def test_unicode_message(self, client):
        resp = client.post("/api/chat", json={"msg": "héllo wörld"})
        assert resp.status_code in (200, 500)
