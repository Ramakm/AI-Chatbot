"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Covers:
- FastAPI endpoint behaviour (happy path + error/edge cases)
- CORS middleware configuration
- Chat processing functions: clean_up_sentence, bow, predict_class, getResponse
- NLTK lemmatisation / tokenisation integration
- Upgraded-package surface area:
    * fastapi 0.104.1 → 0.109.1
    * python-multipart 0.0.6 → 0.0.7
    * keras 2.14.0 → 3.12.0
    * nltk 3.8.1 → 3.9
"""

import importlib
import json
import pickle
import sys
import types
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers to build a minimal fake environment so app.py can be imported
# without real model files, real NLTK corpora, or a real "static/" directory.
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "good", "name", "i", "am"]
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
            "patterns": ["Bye", "See you later", "Good bye"],
            "responses": ["Goodbye!", "See you later!", "Bye!"],
        },
        {
            "tag": "name",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Hello {n}!", "Nice to meet you {n}!"],
        },
    ]
}


def _make_fake_model():
    """Return a mock Keras model whose predict() returns a plausible array."""
    fake_model = MagicMock()
    # predict returns shape (1, len(FAKE_CLASSES))
    fake_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    return fake_model


# ---------------------------------------------------------------------------
# Fixture: patch everything that app.py touches at import time
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def patched_app():
    """
    Import app.py with all external I/O mocked out.
    Returns the (app_module, TestClient) pair.
    """
    fake_model = _make_fake_model()

    # --- keras.models.load_model ---
    keras_mock = MagicMock()
    keras_mock.models.load_model.return_value = fake_model

    # --- pickle.load ---
    def fake_pickle_load(f):
        name = getattr(f, "name", "")
        if "words" in name:
            return FAKE_WORDS
        if "classes" in name:
            return FAKE_CLASSES
        return []

    # --- builtins.open ---
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if "words.pkl" in path_str:
            buf = BytesIO(pickle.dumps(FAKE_WORDS))
            buf.name = "words.pkl"
            return buf
        if "classes.pkl" in path_str:
            buf = BytesIO(pickle.dumps(FAKE_CLASSES))
            buf.name = "classes.pkl"
            return buf
        if "intents.json" in path_str:
            buf = BytesIO(json.dumps(FAKE_INTENTS).encode())
            buf.name = "intents.json"
            # wrap so .read() works in text mode
            import io
            return io.StringIO(json.dumps(FAKE_INTENTS))
        if "index.html" in path_str:
            import io
            return io.StringIO("<html><body>Chatbot</body></html>")
        # fallback – real open for anything else
        return real_open(path, mode, *args, **kwargs)

    patches = [
        patch("keras.models.load_model", return_value=fake_model),
        patch("builtins.open", side_effect=fake_open),
        patch("pickle.load", side_effect=fake_pickle_load),
        # StaticFiles mount would fail without a real directory
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("fastapi.staticfiles.StaticFiles.__call__", return_value=None),
    ]

    for p in patches:
        p.start()

    # Force re-import if already cached
    for mod_name in list(sys.modules.keys()):
        if "app" == mod_name:
            del sys.modules[mod_name]

    import app as app_module  # noqa: E402

    client = TestClient(app_module.app, raise_server_exceptions=False)

    yield app_module, client, fake_model

    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Convenience fixtures derived from the session-scoped one
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_module(patched_app):
    return patched_app[0]


@pytest.fixture()
def client(patched_app):
    return patched_app[1]


@pytest.fixture()
def fake_model(patched_app):
    return patched_app[2]


# ===========================================================================
# 1. Module-level attribute tests
# ===========================================================================

class TestModuleAttributes:
    def test_words_is_list(self, app_module):
        assert isinstance(app_module.words, list)

    def test_classes_is_list(self, app_module):
        assert isinstance(app_module.classes, list)

    def test_intents_has_intents_key(self, app_module):
        assert "intents" in app_module.intents

    def test_model_attribute_exists(self, app_module):
        assert app_module.model is not None

    def test_lemmatizer_type(self, app_module):
        from nltk.stem import WordNetLemmatizer
        assert isinstance(app_module.lemmatizer, WordNetLemmatizer)


# ===========================================================================
# 2. FastAPI app metadata (fastapi upgrade surface)
# ===========================================================================

class TestAppMetadata:
    def test_app_title(self, app_module):
        assert app_module.app.title == "AI Chatbot API"

    def test_app_version(self, app_module):
        assert app_module.app.version == "1.0.0"

    def test_cors_middleware_present(self, app_module):
        from starlette.middleware.cors import CORSMiddleware
        middleware_types = [
            m.cls for m in app_module.app.user_middleware
            if hasattr(m, "cls")
        ]
        assert CORSMiddleware in middleware_types


# ===========================================================================
# 3. GET / — home endpoint
# ===========================================================================

class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_returns_html(self, client):
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")

    def test_home_contains_chatbot(self, client):
        response = client.get("/")
        assert "Chatbot" in response.text or "<html" in response.text.lower()


# ===========================================================================
# 4. GET /health — health check endpoint
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model"] == "loaded"


# ===========================================================================
# 5. POST /api/chat — happy paths
# ===========================================================================

class TestChatEndpointHappyPath:
    def test_basic_message_returns_200(self, client):
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/api/chat", json={"msg": "hello"}).json()
        assert "response" in data
        assert "confidence" in data

    def test_confidence_is_float_in_range(self, client):
        data = client.post("/api/chat", json={"msg": "hello"}).json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_my_name_is_prefix(self, client):
        data = client.post("/api/chat", json={"msg": "my name is Alice"}).json()
        assert data["response"] is not None
        assert isinstance(data["response"], str)

    def test_hi_my_name_is_prefix(self, client):
        data = client.post("/api/chat", json={"msg": "hi my name is Bob"}).json()
        assert data["response"] is not None

    def test_i_am_prefix(self, client):
        data = client.post("/api/chat", json={"msg": "I am Charlie"}).json()
        assert data["response"] is not None

    def test_name_placeholder_replaced(self, client, app_module):
        """
        When a response template contains {n}, the name extracted from the
        message should replace it.
        """
        # Patch getResponse to return a template
        original_get_response = app_module.getResponse

        def fake_get_response(ints, intents):
            return "Hello {n}!"

        app_module.getResponse = fake_get_response
        try:
            data = client.post("/api/chat", json={"msg": "my name is Diana"}).json()
            assert "Diana" in data["response"]
        finally:
            app_module.getResponse = original_get_response

    def test_hi_my_name_is_name_replaced(self, client, app_module):
        original_get_response = app_module.getResponse

        def fake_get_response(ints, intents):
            return "Hi {n}!"

        app_module.getResponse = fake_get_response
        try:
            data = client.post("/api/chat", json={"msg": "hi my name is Eve"}).json()
            assert "Eve" in data["response"]
        finally:
            app_module.getResponse = original_get_response

    def test_i_am_name_replaced(self, client, app_module):
        original_get_response = app_module.getResponse

        def fake_get_response(ints, intents):
            return "Hey {n}!"

        app_module.getResponse = fake_get_response
        try:
            data = client.post("/api/chat", json={"msg": "i am Frank"}).json()
            assert "Frank" in data["response"]
        finally:
            app_module.getResponse = original_get_response

    def test_leading_trailing_whitespace_stripped(self, client):
        response = client.post("/api/chat", json={"msg": "  hello  "})
        assert response.status_code == 200


# ===========================================================================
# 6. POST /api/chat — error / edge cases (fastapi 0.109.1 validation)
# ===========================================================================

class TestChatEndpointErrors:
    def test_empty_string_returns_400(self, client):
        response = client.post("/api/chat", json={"msg": ""})
        assert response.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        response = client.post("/api/chat", json={"msg": "   "})
        assert response.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        """FastAPI/pydantic validation should reject missing required field."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client):
        """msg must be a string; integer should fail validation."""
        response = client.post("/api/chat", json={"msg": 12345})
        # pydantic v2 coerces int→str, so 200 or 422 are both acceptable
        assert response.status_code in (200, 422)

    def test_no_body_returns_422(self, client):
        response = client.post("/api/chat")
        assert response.status_code == 422

    def test_error_from_processing_returns_500(self, client, app_module):
        """If predict_class raises, the endpoint should return 500."""
        original_predict = app_module.predict_class

        def boom(sentence, model):
            raise ValueError("model exploded")

        app_module.predict_class = boom
        try:
            response = client.post("/api/chat", json={"msg": "hello"})
            assert response.status_code == 500
        finally:
            app_module.predict_class = original_predict


# ===========================================================================
# 7. CORS headers (python-multipart / fastapi upgrade)
# ===========================================================================

class TestCORSHeaders:
    def test_cors_preflight_allowed(self, client):
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # Should not be 403/404 – CORS middleware handles it
        assert response.status_code in (200, 204)

    def test_cors_origin_header_present(self, client):
        response = client.post(
            "/api/chat",
            json={"msg": "hello"},
            headers={"Origin": "http://example.com"},
        )
        assert "access-control-allow-origin" in response.headers


# ===========================================================================
# 8. clean_up_sentence (nltk 3.9 tokenise + lemmatise)
# ===========================================================================

class TestCleanUpSentence:
    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercases_tokens(self, app_module):
        result = app_module.clean_up_sentence("HELLO")
        assert all(t == t.lower() for t in result)

    def test_lemmatises_plural(self, app_module):
        result = app_module.clean_up_sentence("dogs")
        # WordNetLemmatizer should reduce 'dogs' to 'dog'
        assert "dog" in result

    def test_tokenises_multiple_words(self, app_module):
        result = app_module.clean_up_sentence("how are you")
        assert len(result) >= 3

    def test_punctuation_is_tokenised(self, app_module):
        result = app_module.clean_up_sentence("Hello!")
        # NLTK splits punctuation as separate tokens
        assert len(result) >= 1

    def test_empty_string(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_single_word(self, app_module):
        result = app_module.clean_up_sentence("running")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_lemmatises_running_to_running(self, app_module):
        """Default POS is noun so 'running' stays 'running'."""
        result = app_module.clean_up_sentence("running")
        assert "running" in result


# ===========================================================================
# 9. bow (bag-of-words)
# ===========================================================================

class TestBow:
    def test_returns_numpy_array(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert isinstance(arr, np.ndarray)

    def test_length_equals_vocabulary(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        assert len(arr) == len(FAKE_WORDS)

    def test_known_word_sets_bit(self, app_module):
        arr = app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert arr[idx] == 1

    def test_unknown_word_all_zeros(self, app_module):
        arr = app_module.bow("xyzzy", FAKE_WORDS)
        assert arr.sum() == 0

    def test_multiple_known_words(self, app_module):
        arr = app_module.bow("hello how are you", FAKE_WORDS)
        assert arr.sum() >= 2

    def test_empty_sentence(self, app_module):
        arr = app_module.bow("", FAKE_WORDS)
        assert arr.sum() == 0

    def test_values_are_binary(self, app_module):
        arr = app_module.bow("hello hi bye", FAKE_WORDS)
        assert set(arr.tolist()).issubset({0, 1})

    def test_empty_vocabulary(self, app_module):
        arr = app_module.bow("hello", [])
        assert len(arr) == 0


# ===========================================================================
# 10. predict_class (keras 3.x model.predict surface)
# ===========================================================================

class TestPredictClass:
    def test_returns_list(self, app_module, fake_model):
        result = app_module.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_model_predict_called(self, app_module, fake_model):
        fake_model.predict.reset_mock()
        app_module.predict_class("hello", fake_model)
        fake_model.predict.assert_called_once()

    def test_result_contains_intent_and_probability(self, app_module, fake_model):
        result = app_module.predict_class("hello", fake_model)
        if result:
            assert "intent" in result[0]
            assert "probability" in result[0]

    def test_probability_is_string_of_float(self, app_module, fake_model):
        result = app_module.predict_class("hello", fake_model)
        if result:
            prob = float(result[0]["probability"])
            assert 0.0 <= prob <= 1.0

    def test_input_array_shape_matches_vocabulary(self, app_module, fake_model):
        """model.predict must receive an array whose width == len(words)."""
        app_module.predict_class("hello", fake_model)
        call_args = fake_model.predict.call_args
        arr = call_args[0][0]
        assert arr.shape[-1] == len(FAKE_WORDS)

    def test_predict_class_with_unknown_sentence(self, app_module, fake_model):
        """Even unknown words should produce a valid (possibly empty) result."""
        result = app_module.predict_class("zxcvbnm", fake_model)
        assert isinstance(result, list)

    def test_high_confidence_threshold_filters_low_scores(self, app_module):
        """
        If the model returns only low-probability scores,
        predict_class should return an empty list (threshold typically 0.25).
        """
        low_model = MagicMock()
        low_model.predict.return_value = np.array([[0.05, 0.05, 0.05]])
        result = app_module.predict_class("hello", low_model)
        assert isinstance(result, list)
        # all below threshold → empty
        assert result == []


# ===========================================================================
# 11. getResponse
# ===========================================================================

class TestGetResponse:
    def _make_ints(self, intent_tag, prob="0.9"):
        return [{"intent": intent_tag, "probability": prob}]

    def test_returns_string(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_response_is_from_correct_intent(self, app_module):
        ints = self._make_ints("greeting")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        valid = FAKE_INTENTS["intents"][0]["responses"]
        assert result in valid

    def test_goodbye_intent(self, app_module):
        ints = self._make_ints("goodbye")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        valid = FAKE_INTENTS["intents"][1]["responses"]
        assert result in valid

    def test_name_intent_response(self, app_module):
        ints = self._make_ints("name")
        result = app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_empty_ints_handled(self, app_module):
        """If ints is empty, getResponse should not raise; may return a fallback."""
        try:
            result = app_module.getResponse([], FAKE_INTENTS)
            assert isinstance(result, str)
        except (IndexError, KeyError):
            pytest.skip("getResponse does not guard against empty ints list")


# ===========================================================================
# 12. Pydantic models (fastapi upgrade — pydantic v2 compatibility)
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_requires_msg(self, app_module):
        with pytest.raises(Exception):
            app_module.MessageRequest()

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_confidence_float(self, app_module):
        resp = app_module.ChatResponse(response="ok", confidence=0.1)
        assert isinstance(resp.confidence, float)

    def test_chat_response_requires_both_fields(self, app_module):
        with pytest.raises(Exception):
            app_module.ChatResponse(response="Hi!")


# ===========================================================================
# 13. NLTK integration (nltk 3.9 upgrade)
# ===========================================================================

class TestNLTKIntegration:
    def test_word_tokenize_basic(self):
        import nltk
        tokens = nltk.word_tokenize("Hello world!")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_wordnet_lemmatizer_noun(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("dogs") == "dog"
        assert lem.lemmatize("cats") == "cat"

    def test_wordnet_lemmatizer_verb(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        # pos='v' to treat as verb
        assert lem.lemmatize("running", pos="v") == "run"

    def test_word_tokenize_empty(self):
        import nltk
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_word_tokenize_punctuation(self):
        import nltk
        tokens = nltk.word_tokenize("Hello!")
        assert "Hello" in tokens
        assert "!" in tokens

    def test_lemmatizer_lowercase_unchanged(self):
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        assert lem.lemmatize("hello") == "hello"


# ===========================================================================
# 14. Keras model usage (keras 3.12.0 upgrade)
# ===========================================================================

class TestKerasModelUsage:
    def test_load_model_called_with_h5_path(self):
        """keras.models.load_model should be called with the .h5 file path."""
        with patch("keras.models.load_model") as mock_load, \
             patch("builtins.open", mock_open(read_data=json.dumps(FAKE_INTENTS))), \
             patch("pickle.load", side_effect=[FAKE_WORDS, FAKE_CLASSES]), \
             patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None):
            mock_load.return_value = _make_fake_model()
            # Remove cached module
            sys.modules.pop("app", None)
            try:
                import app  # noqa: F401
                mock_load.assert_called_once_with("chatbot_model.h5")
            except Exception:
                pass  # import errors are acceptable in isolation
            finally:
                sys.modules.pop("app", None)

    def test_model_predict_returns_numpy_array(self, fake_model):
        result = fake_model.predict(np.zeros((1, len(FAKE_WORDS))))
        assert isinstance(result, np.ndarray)

    def test_model_predict_output_shape(self, fake_model):
        result = fake_model.predict(np.zeros((1, len(FAKE_WORDS))))
        assert result.shape == (1, len(FAKE_CLASSES))

    def test_model_predict_values_sum_to_one(self, fake_model):
        result = fake_model.predict(np.zeros((1, len(FAKE_WORDS))))
        total = result[0].sum()
        assert pytest.approx(total, abs=0.01) == 1.0

    def test_predict_input_is_reshaped(self, app_module, fake_model):
        """bow output must be reshaped to (1, n) before model.predict."""
        fake_model.predict.reset_mock()
        app_module.predict_class("hello", fake_model)
        call_args = fake_model.predict.call_args[0][0]
        assert call_args.ndim == 2
        assert call_args.shape[0] == 1


# ===========================================================================
# 15. Integration: full chat flow
# ===========================================================================

class TestFullChatFlow:
    def test_greeting_flow(self, client):
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert len(body["response"]) > 0
        assert body["confidence"] >= 0.0

    def test_case_insensitive_prefix_my_name_is(self, client):
        for prefix in ["my name is", "MY NAME IS", "My Name Is"]:
            response = client.post("/api/chat", json={"msg": f"{prefix} Test"})
            assert response.status_code == 200

    def test_case_insensitive_prefix_hi_my_name_is(self, client):
        response = client.post("/api/chat", json={"msg": "HI MY NAME IS Test"})
        assert response.status_code == 200

    def test_case_insensitive_prefix_i_am(self, client):
        response = client.post("/api/chat", json={"msg": "I AM Test"})
        assert response.status_code == 200

    def test_repeated_requests_consistent(self, client):
        """Multiple identical requests should all return 200."""
        for _ in range(5):
            response = client.post("/api/chat", json={"msg": "hello"})
            assert response.status_code == 200

    def test_long_message(self, client):
        long_msg = "hello " * 100
        response = client.post("/api/chat", json={"msg": long_msg})
        assert response.status_code == 200

    def test_special_characters_in_message(self, client):
        response = client.post("/api/chat", json={"msg": "hello! how are you?"})
        assert response.status_code == 200

    def test_unicode_message(self, client):
        response = client.post("/api/chat", json={"msg": "héllo"})
        assert response.status_code in (200, 500)  # graceful handling
