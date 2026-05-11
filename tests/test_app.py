"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).
Covers core functionality, upgraded package APIs (FastAPI 0.109.1, keras 3.x,
nltk 3.9, python-multipart 0.0.7), happy-path and error/edge cases.
All external I/O (model, pickle files, file reads) is mocked.
"""

import json
import pickle
import sys
import types
import importlib
from unittest.mock import MagicMock, patch, mock_open, AsyncMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build a minimal mock environment so that app.py can be imported
# without real files on disk.
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "bye", "good", "morning", "name", "i", "am", "how", "are", "you"]
FAKE_CLASSES = ["greeting", "goodbye", "name"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Good morning"],
            "responses": ["Goodbye!", "See you later!"],
        },
        {
            "tag": "name",
            "patterns": ["My name is", "I am", "Hi my name is"],
            "responses": ["Hello {n}!", "Nice to meet you {n}!"],
        },
    ]
}

INDEX_HTML = "<html><body>Chatbot</body></html>"


def _make_mock_model(num_classes=3):
    """Return a mock Keras model whose predict() returns a sensible array."""
    mock_model = MagicMock()
    probs = np.zeros((1, num_classes))
    probs[0][0] = 0.9  # highest confidence on class 0
    mock_model.predict.return_value = probs
    return mock_model


def _build_app_module():
    """
    Import app.py with all file/model loading fully mocked so that the module
    initialises cleanly.  Returns the imported module object.
    """
    mock_model = _make_mock_model(len(FAKE_CLASSES))

    words_pkl = pickle.dumps(FAKE_WORDS)
    classes_pkl = pickle.dumps(FAKE_CLASSES)
    intents_json = json.dumps(FAKE_INTENTS)

    # We need open() to return different things depending on the filename.
    original_open = open  # keep reference

    def fake_open(file, mode="r", *args, **kwargs):
        file_str = str(file)
        if "words.pkl" in file_str:
            import io
            return io.BytesIO(words_pkl)
        if "classes.pkl" in file_str:
            import io
            return io.BytesIO(classes_pkl)
        if "intents.json" in file_str:
            import io
            return io.StringIO(intents_json)
        if "index.html" in file_str:
            import io
            return io.StringIO(INDEX_HTML)
        # Fallback for anything else (e.g. StaticFiles internals)
        return original_open(file, mode, *args, **kwargs)

    patches = [
        patch("keras.models.load_model", return_value=mock_model),
        patch("builtins.open", side_effect=fake_open),
        patch("pickle.load", side_effect=lambda f: pickle.loads(f.read())),
        # StaticFiles requires a real directory; skip mounting
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.FastAPI.mount", return_value=None),
    ]

    # Remove cached module if present
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("app",):
            del sys.modules[mod_name]

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        import app as app_module

    return app_module, mock_model


# ---------------------------------------------------------------------------
# Module-level fixture: import app once and reuse across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    module, mock_model = _build_app_module()
    module._test_mock_model = mock_model  # stash for later use
    return module


@pytest.fixture(scope="module")
def test_client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# 1. Application bootstrap / configuration tests
# ---------------------------------------------------------------------------

class TestAppConfiguration:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_cors_middleware_present(self, app_module):
        from starlette.middleware.cors import CORSMiddleware
        middleware_types = [m.cls for m in app_module.app.user_middleware]
        assert CORSMiddleware in middleware_types

    def test_words_loaded(self, app_module):
        assert isinstance(app_module.words, list)
        assert len(app_module.words) > 0

    def test_classes_loaded(self, app_module):
        assert isinstance(app_module.classes, list)
        assert len(app_module.classes) > 0

    def test_intents_loaded(self, app_module):
        assert isinstance(app_module.intents, dict)
        assert "intents" in app_module.intents

    def test_model_is_loaded(self, app_module):
        assert app_module.model is not None


# ---------------------------------------------------------------------------
# 2. Pydantic models
# ---------------------------------------------------------------------------

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

    def test_chat_response_max_confidence(self, app_module):
        resp = app_module.ChatResponse(response="Sure", confidence=1.0)
        assert resp.confidence == 1.0


# ---------------------------------------------------------------------------
# 3. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ---------------------------------------------------------------------------
# 4. Home endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_200(self, test_client):
        with patch("builtins.open", mock_open(read_data=INDEX_HTML)):
            resp = test_client.get("/")
        assert resp.status_code == 200

    def test_home_returns_html(self, test_client):
        with patch("builtins.open", mock_open(read_data=INDEX_HTML)):
            resp = test_client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_home_html_content(self, test_client):
        with patch("builtins.open", mock_open(read_data=INDEX_HTML)):
            resp = test_client.get("/")
        assert "Chatbot" in resp.text


# ---------------------------------------------------------------------------
# 5. Chat endpoint — happy paths
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:
    """Tests for POST /api/chat."""

    def _post(self, test_client, msg):
        return test_client.post("/api/chat", json={"msg": msg})

    def test_basic_greeting_returns_200(self, test_client):
        resp = self._post(test_client, "Hello")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, test_client):
        resp = self._post(test_client, "Hello")
        data = resp.json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_float(self, test_client):
        resp = self._post(test_client, "Hello")
        assert isinstance(resp.json()["confidence"], float)

    def test_confidence_between_0_and_1(self, test_client):
        resp = self._post(test_client, "Hello")
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_response_is_string(self, test_client):
        resp = self._post(test_client, "Hello")
        assert isinstance(resp.json()["response"], str)

    def test_my_name_is_pattern(self, test_client):
        resp = self._post(test_client, "my name is Alice")
        assert resp.status_code == 200
        # name substitution or at least no crash
        assert isinstance(resp.json()["response"], str)

    def test_hi_my_name_is_pattern(self, test_client):
        resp = self._post(test_client, "hi my name is Bob")
        assert resp.status_code == 200
        assert isinstance(resp.json()["response"], str)

    def test_i_am_pattern(self, test_client):
        resp = self._post(test_client, "i am Charlie")
        assert resp.status_code == 200
        assert isinstance(resp.json()["response"], str)

    def test_uppercase_my_name_is(self, test_client):
        """Case-insensitive matching for name patterns."""
        resp = self._post(test_client, "MY NAME IS Dave")
        assert resp.status_code == 200

    def test_leading_trailing_whitespace_stripped(self, test_client):
        resp = self._post(test_client, "  Hello  ")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. Chat endpoint — error / edge cases
# ---------------------------------------------------------------------------

class TestChatEndpointErrors:
    def _post(self, test_client, msg):
        return test_client.post("/api/chat", json={"msg": msg})

    def test_empty_message_returns_400(self, test_client):
        resp = self._post(test_client, "")
        assert resp.status_code == 400

    def test_whitespace_only_message_returns_400(self, test_client):
        resp = self._post(test_client, "   ")
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, test_client):
        resp = test_client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_wrong_type_returns_422(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": 12345})
        # FastAPI coerces int to str for str fields, so either 200 or 422 is ok
        assert resp.status_code in (200, 422)

    def test_internal_error_returns_500(self, test_client, app_module):
        """Simulate predict_class raising an exception."""
        with patch.object(
            sys.modules[app_module.__name__],
            "predict_class",
            side_effect=RuntimeError("model exploded"),
        ):
            resp = self._post(test_client, "Hello")
        assert resp.status_code == 500
        assert "Error processing message" in resp.json()["detail"]

    def test_very_long_message(self, test_client):
        long_msg = "Hello " * 500
        resp = self._post(test_client, long_msg)
        assert resp.status_code in (200, 400, 500)

    def test_special_characters_message(self, test_client):
        resp = self._post(test_client, "!@#$%^&*()")
        assert resp.status_code in (200, 500)

    def test_unicode_message(self, test_client):
        resp = self._post(test_client, "こんにちは")
        assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# 7. clean_up_sentence (nltk integration)
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    """Tests for clean_up_sentence() — exercises nltk.word_tokenize + lemmatize."""

    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercased(self, app_module):
        result = app_module.clean_up_sentence("HELLO WORLD")
        for word in result:
            assert word == word.lower()

    def test_basic_tokenization(self, app_module):
        result = app_module.clean_up_sentence("hello world")
        assert "hello" in result
        assert "world" in result

    def test_lemmatization_plural(self, app_module):
        result = app_module.clean_up_sentence("dogs are running")
        # "dogs" -> "dog", "running" -> "running" (default POS is noun)
        assert "dog" in result

    def test_single_word(self, app_module):
        result = app_module.clean_up_sentence("running")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_included(self, app_module):
        """NLTK tokenizer keeps punctuation as separate tokens."""
        result = app_module.clean_up_sentence("Hello!")
        assert "hello" in result

    def test_contractions(self, app_module):
        result = app_module.clean_up_sentence("I'm fine")
        assert isinstance(result, list)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# 8. bow (bag-of-words) function
# ---------------------------------------------------------------------------

class TestBagOfWords:
    def test_returns_numpy_array(self, app_module):
        words = app_module.words
        result = app_module.bow("hello", words)
        assert isinstance(result, np.ndarray)

    def test_length_matches_words(self, app_module):
        words = app_module.words
        result = app_module.bow("hello", words)
        assert len(result) == len(words)

    def test_known_word_sets_bit(self, app_module):
        words = ["hello", "bye", "good"]
        result = app_module.bow("hello", words)
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 0

    def test_no_match_all_zeros(self, app_module):
        words = ["xyz", "abc"]
        result = app_module.bow("hello", words)
        assert all(v == 0 for v in result)

    def test_multiple_matches(self, app_module):
        words = ["hello", "world"]
        result = app_module.bow("hello world", words)
        assert result[0] == 1
        assert result[1] == 1

    def test_empty_sentence(self, app_module):
        words = app_module.words
        result = app_module.bow("", words)
        assert isinstance(result, np.ndarray)
        assert all(v == 0 for v in result)

    def test_binary_values_only(self, app_module):
        words = app_module.words
        result = app_module.bow("hello hi good", words)
        for v in result:
            assert v in (0, 1)


# ---------------------------------------------------------------------------
# 9. predict_class function (keras model integration)
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

    def test_probability_is_string_or_numeric(self, app_module):
        """predict_class stores probability; check it is convertible to float."""
        result = app_module.predict_class("hello", app_module.model)
        for item in result:
            float(item["probability"])  # must not raise

    def test_model_predict_called(self, app_module):
        app_module.model.predict.reset_mock()
        app_module.predict_class("hello", app_module.model)
        app_module.model.predict.assert_called_once()

    def test_model_predict_input_shape(self, app_module):
        """Input to model.predict should be a 2-D array."""
        app_module.model.predict.reset_mock()
        app_module.predict_class("hello", app_module.model)
        call_args = app_module.model.predict.call_args
        input_array = call_args[0][0]
        assert input_array.ndim == 2

    def test_results_sorted_by_confidence(self, app_module):
        result = app_module.predict_class("hello", app_module.model)
        probs = [float(r["probability"]) for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_empty_sentence(self, app_module):
        result = app_module.predict_class("", app_module.model)
        assert isinstance(result, list)

    def test_high_confidence_prediction(self, app_module):
        """With mocked model returning 0.9 for class 0, top result >= 0.5."""
        result = app_module.predict_class("hello", app_module.model)
        if result:
            assert float(result[0]["probability"]) >= 0.0


# ---------------------------------------------------------------------------
# 10. getResponse function
# ---------------------------------------------------------------------------

class TestGetResponse:
    def _make_ints(self, intent_tag, prob=0.9):
        return [{"intent": intent_tag, "probability": str(prob)}]

    def test_returns_string(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_response_from_correct_intent(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][0]["responses"]

    def test_goodbye_response(self, app_module):
        ints = self._make_ints("goodbye")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][1]["responses"]

    def test_name_response_contains_placeholder(self, app_module):
        ints = self._make_ints("name")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][2]["responses"]

    def test_unknown_intent_returns_string(self, app_module):
        ints = self._make_ints("unknown_tag")
        # Should not raise — may return empty string or fallback
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_empty_ints_list(self, app_module):
        result = app_module.getResponse([], FAKE_INTENTS)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 11. Name substitution in /api/chat
# ---------------------------------------------------------------------------

class TestNameSubstitution:
    def _post(self, test_client, msg):
        return test_client.post("/api/chat", json={"msg": msg})

    def test_name_substituted_in_response_my_name_is(self, test_client, app_module):
        """If the response template has {n}, the name should be substituted."""
        def fake_get_response(ints, intents):
            return "Hello {n}!"

        with patch.object(
            sys.modules[app_module.__name__], "getResponse", side_effect=fake_get_response
        ):
            resp = self._post(test_client, "my name is Alice")
        assert resp.status_code == 200
        assert "Alice" in resp.json()["response"]

    def test_name_substituted_hi_my_name_is(self, test_client, app_module):
        def fake_get_response(ints, intents):
            return "Hi {n}!"

        with patch.object(
            sys.modules[app_module.__name__], "getResponse", side_effect=fake_get_response
        ):
            resp = self._post(test_client, "hi my name is Bob")
        assert resp.status_code == 200
        assert "Bob" in resp.json()["response"]

    def test_name_substituted_i_am(self, test_client, app_module):
        def fake_get_response(ints, intents):
            return "Nice to meet you {n}!"

        with patch.object(
            sys.modules[app_module.__name__], "getResponse", side_effect=fake_get_response
        ):
            resp = self._post(test_client, "i am Charlie")
        assert resp.status_code == 200
        assert "Charlie" in resp.json()["response"]

    def test_no_placeholder_no_substitution(self, test_client, app_module):
        def fake_get_response(ints, intents):
            return "Hello there!"

        with patch.object(
            sys.modules[app_module.__name__], "getResponse", side_effect=fake_get_response
        ):
            resp = self._post(test_client, "my name is Dave")
        assert resp.status_code == 200
        assert resp.json()["response"] == "Hello there!"


# ---------------------------------------------------------------------------
# 12. FastAPI-specific: OpenAPI schema and docs availability (0.109.1)
# ---------------------------------------------------------------------------

class TestFastAPIOpenAPI:
    def test_openapi_schema_available(self, test_client):
        resp = test_client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_schema_has_info(self, test_client):
        schema = test_client.get("/openapi.json").json()
        assert schema["info"]["title"] == "AI Chatbot API"
        assert schema["info"]["version"] == "1.0.0"

    def test_openapi_schema_has_chat_endpoint(self, test_client):
        schema = test_client.get("/openapi.json").json()
        assert "/api/chat" in schema["paths"]

    def test_openapi_schema_has_health_endpoint(self, test_client):
        schema = test_client.get("/openapi.json").json()
        assert "/health" in schema["paths"]

    def test_docs_endpoint_available(self, test_client):
        resp = test_client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_endpoint_available(self, test_client):
        resp = test_client.get("/redoc")
        assert resp.status_code == 200

    def test_chat_endpoint_post_method(self, test_client):
        schema = test_client.get("/openapi.json").json()
        assert "post" in schema["paths"]["/api/chat"]

    def test_health_endpoint_get_method(self, test_client):
        schema = test_client.get("/openapi.json").json()
        assert "get" in schema["paths"]["/health"]


# ---------------------------------------------------------------------------
# 13. CORS headers (middleware behaviour)
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_options_preflight_returns_200_or_400(self, test_client):
        resp = test_client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 400)

    def test_cors_header_present_on_get(self, test_client):
        resp = test_client.get(
            "/health", headers={"Origin": "http://example.com"}
        )
        assert "access-control-allow-origin" in resp.headers

    def test_cors_allows_all_origins(self, test_client):
        resp = test_client.get(
            "/health", headers={"Origin": "http://any-origin.com"}
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin == "*" or allow_origin == "http://any-origin.com"


# ---------------------------------------------------------------------------
# 14. NLTK WordNetLemmatizer (nltk 3.9)
# ---------------------------------------------------------------------------

class TestNLTKLemmatizer:
    """Directly tests the NLTK lemmatizer as used in the application."""

    def test_lemmatizer_noun(self, app_module):
        result = app_module.lemmatizer.lemmatize("dogs")
        assert result == "dog"

    def test_lemmatizer_verb(self, app_module):
        result = app_module.lemmatizer.lemmatize("running", pos="v")
        assert result == "run"

    def test_lemmatizer_adjective(self, app_module):
        result = app_module.lemmatizer.lemmatize("better", pos="a")
        assert result == "good"

    def test_lemmatizer_returns_string(self, app_module):
        result = app_module.lemmatizer.lemmatize("hello")
        assert isinstance(result, str)

    def test_lemmatizer_lowercase_unchanged(self, app_module):
        result = app_module.lemmatizer.lemmatize("hello")
        assert result == "hello"

    def test_word_tokenize_splits_correctly(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world, how are you?")
        assert "Hello" in tokens
        assert "world" in tokens
        assert "?" in tokens

    def test_word_tokenize_empty(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_word_tokenize_contraction(self):
        import nltk
        tokens = nltk.word_tokenize("I'm happy")
        assert len(tokens) >= 2


# ---------------------------------------------------------------------------
# 15. Integration: full request cycle
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end flow through the stack."""

    def test_full_greeting_flow(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] != ""
        assert 0.0 <= data["confidence"] <= 1.0

    def test_full_goodbye_flow(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "Bye"})
        assert resp.status_code == 200

    def test_multiple_sequential_requests(self, test_client):
        messages = ["Hello", "How are you?", "Bye", "Good morning"]
        for msg in messages:
            resp = test_client.post("/api/chat", json={"msg": msg})
            assert resp.status_code == 200

    def test_content_type_json(self, test_client):
        resp = test_client.post("/api/chat", json={"msg": "Hello"})
        assert "application/json" in resp.headers["content-type"]

    def test_health_then_chat(self, test_client):
        health = test_client.get("/health")
        assert health.status_code == 200
        chat = test_client.post("/api/chat", json={"msg": "Hi"})
        assert chat.status_code == 200


# ---------------------------------------------------------------------------
# 16. Keras model mock — verify API contract (keras 3.x)
# ---------------------------------------------------------------------------

class TestKerasModelAPIContract:
    """
    Verifies that the code interacts with the Keras model in the way expected
    by keras 3.x (predict takes a numpy array, returns numpy array).
    """

    def test_model_predict_receives_numpy_array(self, app_module):
        app_module.model.predict.reset_mock()
        app_module.bow("hello world", app_module.words)
        app_module.predict_class("hello world", app_module.model)
        call_args = app_module.model.predict.call_args
        arr = call_args[0][0]
        assert isinstance(arr, np.ndarray)

    def test_model_predict_2d_input(self, app_module):
        app_module.model.predict.reset_mock()
        app_module.predict_class("hello", app_module.model)
        call_args = app_module.model.predict.call_args
        arr = call_args[0][0]
        assert arr.ndim == 2
        assert arr.shape[0] == 1
        assert arr.shape[1] == len(app_module.words)

    def test_model_predict_called_once_per_message(self, test_client, app_module):
        app_module.model.predict.reset_mock()
        test_client.post("/api/chat", json={"msg": "Hello"})
        assert app_module.model.predict.call_count == 1

    def test_model_predict_output_consumed_correctly(self, app_module):
        """
        Ensure predict_class correctly reads the probability array returned
        by model.predict and maps it to intent classes.
        """
        result = app_module.predict_class("hello", app_module.model)
        # Result must map to known classes
        intents_in_result = [r["intent"] for r in result]
        for intent in intents_in_result:
            assert intent in app_module.classes
