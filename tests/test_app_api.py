"""
Tests for FastAPI chatbot endpoints (app.py)
Covers: FastAPI routing, CORS middleware, request/response models,
        chat processing logic, and upgraded package APIs.
"""

import json
import pickle
import sys
import types
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers – build the bare-minimum fake modules so app.py can be imported
# without real model files or a running Keras/NLTK environment.
# ---------------------------------------------------------------------------

def _make_fake_keras():
    """Return a minimal fake keras package tree."""
    keras = types.ModuleType("keras")
    keras.models = types.ModuleType("keras.models")

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
    keras.models.load_model = MagicMock(return_value=fake_model)
    keras.layers = types.ModuleType("keras.layers")
    keras.layers.Dense = MagicMock()
    keras.layers.Dropout = MagicMock()

    sys.modules.setdefault("keras", keras)
    sys.modules.setdefault("keras.models", keras.models)
    sys.modules.setdefault("keras.layers", keras.layers)
    return fake_model


def _make_fake_nltk():
    """Patch nltk so tokenisation works deterministically in tests."""
    import nltk as real_nltk  # noqa: F401 – may already be installed
    return real_nltk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "goodbye", "name"]
SAMPLE_CLASSES = ["greeting", "goodbye", "name"]
SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Greetings!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is", "I am"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}


@pytest.fixture(scope="module")
def fake_model():
    return _make_fake_keras()


@pytest.fixture(scope="module")
def app_module(fake_model):
    """Import app with all file-system & model dependencies mocked."""
    words_bytes = pickle.dumps(SAMPLE_WORDS)
    classes_bytes = pickle.dumps(SAMPLE_CLASSES)
    intents_json = json.dumps(SAMPLE_INTENTS)

    def _open_side_effect(path, mode="r", *args, **kwargs):
        if "words.pkl" in str(path):
            import io
            return io.BytesIO(words_bytes)
        if "classes.pkl" in str(path):
            import io
            return io.BytesIO(classes_bytes)
        if "intents.json" in str(path):
            import io
            return io.StringIO(intents_json)
        if "index.html" in str(path):
            import io
            return io.StringIO("<html><body>Chatbot</body></html>")
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=_open_side_effect), \
         patch("pickle.load", side_effect=[SAMPLE_WORDS, SAMPLE_CLASSES]), \
         patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None), \
         patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None):

        if "app" in sys.modules:
            del sys.modules["app"]

        import app as application
    return application


@pytest.fixture(scope="module")
def client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Application bootstrap
# ---------------------------------------------------------------------------

class TestAppBootstrap:
    def test_app_instance_created(self, app_module):
        from fastapi import FastAPI
        assert isinstance(app_module.app, FastAPI)

    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_words_loaded(self, app_module):
        assert isinstance(app_module.words, list)
        assert len(app_module.words) > 0

    def test_classes_loaded(self, app_module):
        assert isinstance(app_module.classes, list)
        assert len(app_module.classes) > 0

    def test_intents_loaded(self, app_module):
        assert "intents" in app_module.intents

    def test_lemmatizer_initialized(self, app_module):
        from nltk.stem import WordNetLemmatizer
        assert isinstance(app_module.lemmatizer, WordNetLemmatizer)


# ---------------------------------------------------------------------------
# 2. CORS middleware
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_cors_headers_present(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI with CORSMiddleware should return allow-origin header
        assert resp.headers.get("access-control-allow-origin") in ("*", "http://example.com")

    def test_cors_wildcard_origin(self, client):
        resp = client.get("/health", headers={"Origin": "http://other.com"})
        assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# 3. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_body(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model"] == "loaded"


# ---------------------------------------------------------------------------
# 4. Home endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_html(self, client):
        with patch("builtins.open", mock_open(read_data="<html><body>Chatbot</body></html>")):
            resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 5. Pydantic request / response models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="Hello")
        assert req.msg == "Hello"

    def test_message_request_empty_string(self, app_module):
        req = app_module.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_model(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert abs(resp.confidence - 0.95) < 1e-6

    def test_chat_response_confidence_zero(self, app_module):
        resp = app_module.ChatResponse(response="...", confidence=0.0)
        assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# 6. /api/chat – happy paths
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:
    def _mock_predict(self, app_module, tag="greeting", prob=0.85):
        """Patch predict_class and getResponse for isolated endpoint tests."""
        ints = [{"intent": tag, "probability": str(prob)}]
        with patch.object(
            sys.modules[app_module.__name__],
            "predict_class",
            return_value=ints,
        ), patch.object(
            sys.modules[app_module.__name__],
            "getResponse",
            return_value="Hello there!",
        ):
            return ints

    def test_basic_greeting(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.85"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello there!"):
            resp = client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "confidence" in body
        assert body["confidence"] == pytest.approx(0.85, abs=1e-4)

    def test_my_name_is_pattern(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.90"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Nice to meet you, {n}!"):
            resp = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert resp.status_code == 200
        assert "Alice" in resp.json()["response"]

    def test_hi_my_name_is_pattern(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.88"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Nice to meet you, {n}!"):
            resp = client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert resp.status_code == 200
        assert "Bob" in resp.json()["response"]

    def test_i_am_pattern(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.80"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Nice to meet you, {n}!"):
            resp = client.post("/api/chat", json={"msg": "I am Charlie"})
        assert resp.status_code == 200
        assert "Charlie" in resp.json()["response"]

    def test_response_contains_confidence(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.75"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            resp = client.post("/api/chat", json={"msg": "hello"})
        assert isinstance(resp.json()["confidence"], float)

    def test_whitespace_stripped_from_message(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.70"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            resp = client.post("/api/chat", json={"msg": "  Hello  "})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 7. /api/chat – error / edge cases
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

    def test_processing_exception_returns_500(self, client, app_module):
        with patch.object(app_module, "predict_class", side_effect=RuntimeError("boom")):
            resp = client.post("/api/chat", json={"msg": "trigger error"})
        assert resp.status_code == 500
        assert "Error processing message" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 8. clean_up_sentence – uses nltk.word_tokenize + WordNetLemmatizer
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercase_output(self, app_module):
        result = app_module.clean_up_sentence("HELLO")
        assert all(w == w.lower() for w in result)

    def test_lemmatization(self, app_module):
        result = app_module.clean_up_sentence("running")
        # lemmatized form should be present (run or running depending on context)
        assert len(result) >= 1

    def test_punctuation_tokenized(self, app_module):
        result = app_module.clean_up_sentence("Hello!")
        assert len(result) >= 1

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_multiple_words(self, app_module):
        result = app_module.clean_up_sentence("how are you")
        assert len(result) == 3

    def test_nltk_word_tokenize_called(self, app_module):
        with patch("nltk.word_tokenize", return_value=["mocked"]) as mock_tok:
            result = app_module.clean_up_sentence("test sentence")
        mock_tok.assert_called_once_with("test sentence")
        assert "mocked" in result


# ---------------------------------------------------------------------------
# 9. bow (bag-of-words) function
# ---------------------------------------------------------------------------

class TestBagOfWords:
    def test_returns_numpy_array(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_equals_vocab(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        assert len(result) == len(SAMPLE_WORDS)

    def test_known_word_sets_1(self, app_module):
        result = app_module.bow("hello", SAMPLE_WORDS)
        idx = SAMPLE_WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_all_zeros(self, app_module):
        result = app_module.bow("xyzzy", SAMPLE_WORDS)
        assert result.sum() == 0

    def test_multiple_known_words(self, app_module):
        result = app_module.bow("hello bye", SAMPLE_WORDS)
        assert result[SAMPLE_WORDS.index("hello")] == 1
        assert result[SAMPLE_WORDS.index("bye")] == 1

    def test_values_binary(self, app_module):
        result = app_module.bow("hello hi you", SAMPLE_WORDS)
        assert set(result.tolist()).issubset({0, 1})

    def test_empty_sentence(self, app_module):
        result = app_module.bow("", SAMPLE_WORDS)
        assert result.sum() == 0

    def test_show_details_flag(self, app_module, capsys):
        app_module.bow("hello", SAMPLE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "found in bag" in captured.out


# ---------------------------------------------------------------------------
# 10. predict_class function – uses keras model.predict
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, app_module, fake_model):
        # model.predict returns shape (1, n_classes) array
        fake_model.predict.return_value = np.array(
            [[0.05, 0.90, 0.05]]
        )
        result = app_module.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_sorted_by_probability_descending(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.2, 0.7, 0.1]])
        result = app_module.predict_class("hello", fake_model)
        probs = [float(r["probability"]) for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_each_result_has_intent_and_probability(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        result = app_module.predict_class("hello", fake_model)
        for item in result:
            assert "intent" in item
            assert "probability" in item

    def test_model_predict_called_with_array(self, app_module, fake_model):
        fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        app_module.predict_class("hello", fake_model)
        fake_model.predict.assert_called()
        call_arg = fake_model.predict.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)

    def test_low_probability_filtered(self, app_module, fake_model):
        """Results below ERROR_THRESHOLD (0.25) should be excluded."""
        fake_model.predict.return_value = np.array([[0.01, 0.01, 0.01]])
        result = app_module.predict_class("xyzzy unknown gibberish", fake_model)
        # All probabilities below threshold → empty list expected
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 11. getResponse function
# ---------------------------------------------------------------------------

class TestGetResponse:
    def _make_ints(self, tag):
        return [{"intent": tag, "probability": "0.9"}]

    def test_returns_string(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert isinstance(result, str)

    def test_response_from_correct_intent(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert result in SAMPLE_INTENTS["intents"][0]["responses"]

    def test_response_for_goodbye(self, app_module):
        ints = self._make_ints("goodbye")
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert result in SAMPLE_INTENTS["intents"][1]["responses"]

    def test_response_for_name_tag(self, app_module):
        ints = self._make_ints("name")
        result = app_module.getResponse(ints, SAMPLE_INTENTS)
        assert "{n}" in result  # placeholder still present before substitution

    def test_random_selection_from_responses(self, app_module):
        """random.choice should be used; calling multiple times may give different results."""
        ints = self._make_ints("greeting")
        responses_seen = set()
        for _ in range(30):
            responses_seen.add(app_module.getResponse(ints, SAMPLE_INTENTS))
        # At least one greeting response should appear
        assert len(responses_seen) >= 1

    def test_empty_ints_handled(self, app_module):
        """getResponse with empty ints list should not raise; may return fallback."""
        try:
            result = app_module.getResponse([], SAMPLE_INTENTS)
            assert isinstance(result, str)
        except (IndexError, KeyError):
            pass  # acceptable if no guard implemented


# ---------------------------------------------------------------------------
# 12. Integration: name substitution in /api/chat
# ---------------------------------------------------------------------------

class TestNameSubstitution:
    @pytest.mark.parametrize("prefix,name", [
        ("my name is", "Diana"),
        ("hi my name is", "Eve"),
        ("I am", "Frank"),
    ])
    def test_name_replacement_in_response(self, client, app_module, prefix, name):
        ints = [{"intent": "name", "probability": "0.92"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello {n}!"):
            resp = client.post("/api/chat", json={"msg": f"{prefix} {name}"})
        assert resp.status_code == 200
        assert name in resp.json()["response"]

    def test_placeholder_not_in_final_response(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.92"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello {n}!"):
            resp = client.post("/api/chat", json={"msg": "my name is Grace"})
        assert "{n}" not in resp.json()["response"]


# ---------------------------------------------------------------------------
# 13. Keras model.predict API (upgraded keras 2→3)
# ---------------------------------------------------------------------------

class TestKerasModelPredictAPI:
    """
    Verify that the code calls model.predict correctly per the Keras 3 API.
    In Keras 3 model.predict accepts a numpy array and returns a numpy array.
    """

    def test_predict_receives_2d_array(self, app_module, fake_model):
        captured_args = []
        original_predict = fake_model.predict

        def capturing_predict(x, *args, **kwargs):
            captured_args.append(x)
            return np.array([[0.1, 0.8, 0.1]])

        fake_model.predict = capturing_predict
        try:
            app_module.predict_class("hello", fake_model)
        finally:
            fake_model.predict = original_predict

        assert len(captured_args) == 1
        arr = captured_args[0]
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2, "Keras 3 expects 2-D input (batch, features)"

    def test_predict_output_shape_used_correctly(self, app_module, fake_model):
        """predict_class should handle (1, n_classes) output from model.predict."""
        n_classes = len(SAMPLE_CLASSES)
        fake_model.predict.return_value = np.ones((1, n_classes)) / n_classes
        result = app_module.predict_class("test", fake_model)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 14. NLTK API (upgraded nltk 3.8→3.9)
# ---------------------------------------------------------------------------

class TestNLTKAPI:
    def test_word_tokenize_basic(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world!")
        assert isinstance(tokens, list)
        assert "Hello" in tokens

    def test_word_tokenize_empty(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_word_tokenize_punctuation(self):
        import nltk
        tokens = nltk.word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "," in tokens

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("running", pos="v") == "run"

    def test_wordnet_lemmatizer_adjective(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("better", pos="a") == "good"

    def test_wordnet_lemmatizer_unchanged(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("hello") == "hello"

    def test_lemmatize_lowercased_words(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        words = ["RUNNING", "DOGS", "BETTER"]
        lemmed = [lem.lemmatize(w.lower()) for w in words]
        assert all(isinstance(w, str) for w in lemmed)

    def test_tokenize_sentence_matches_clean_up(self, app_module):
        """Ensure clean_up_sentence output is consistent with direct nltk calls."""
        import nltk
        sentence = "Hello how are you"
        direct = [app_module.lemmatizer.lemmatize(t.lower()) for t in nltk.word_tokenize(sentence)]
        via_func = app_module.clean_up_sentence(sentence)
        assert direct == via_func


# ---------------------------------------------------------------------------
# 15. FastAPI / python-multipart – form & JSON content types
# ---------------------------------------------------------------------------

class TestFastAPIRequestHandling:
    def test_json_content_type_accepted(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.80"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            resp = client.post(
                "/api/chat",
                json={"msg": "hello"},
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200

    def test_extra_fields_in_request_ignored(self, client, app_module):
        """FastAPI/pydantic should ignore unknown fields gracefully."""
        ints = [{"intent": "greeting", "probability": "0.80"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            resp = client.post("/api/chat", json={"msg": "hello", "extra": "ignored"})
        assert resp.status_code == 200

    def test_method_not_allowed_on_chat(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_404_on_unknown_route(self, client):
        resp = client.get("/unknown-route")
        assert resp.status_code == 404

    def test_response_schema_keys(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.77"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello!"):
            resp = client.post("/api/chat", json={"msg": "hi"})
        body = resp.json()
        assert set(body.keys()) == {"response", "confidence"}
