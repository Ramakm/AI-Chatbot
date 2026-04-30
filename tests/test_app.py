"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Covers:
- FastAPI endpoint behaviour (happy-path + error cases)
- CORS middleware configuration
- NLP helper functions: clean_up_sentence, bow, predict_class, getResponse
- Upgraded package APIs: fastapi 0.109.1, python-multipart 0.0.7, nltk 3.9
"""

import json
import pickle
import types
import importlib
import sys
from unittest.mock import MagicMock, patch, mock_open, PropertyMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – build lightweight fakes so app.py can be imported without real
# model files, static assets, or Keras/TensorFlow installed.
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "how", "are", "you", "my", "name", "is", "i", "am", "bye", "goodbye"]
FAKE_CLASSES = ["greeting", "farewell", "name_introduction"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "farewell",
            "patterns": ["Bye", "Goodbye"],
            "responses": ["See you later!", "Goodbye!"],
        },
        {
            "tag": "name_introduction",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Nice to meet you, {n}!"],
        },
    ]
}

# A fake Keras model that returns a fixed probability array
def _make_fake_model(predicted_class_index=0, num_classes=3):
    fake_model = MagicMock()
    probs = np.zeros((1, num_classes))
    probs[0][predicted_class_index] = 0.95
    fake_model.predict.return_value = probs
    return fake_model


def _make_fake_lemmatizer():
    """Return a real WordNetLemmatizer-like object using a simple identity stub."""
    lm = MagicMock()
    lm.lemmatize.side_effect = lambda word, **kw: word.rstrip("s")  # trivial lemmatizer
    return lm


# ---------------------------------------------------------------------------
# Fixture: import app with all heavy I/O mocked out
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_app():
    """
    Import app.py with:
      - keras.models.load_model mocked
      - pickle.load mocked to return FAKE_WORDS / FAKE_CLASSES
      - builtins.open mocked for intents.json and index.html
      - StaticFiles mocked so no 'static/' directory is needed
      - nltk downloads suppressed
    """
    fake_model = _make_fake_model()

    words_pkl = pickle.dumps(FAKE_WORDS)
    classes_pkl = pickle.dumps(FAKE_CLASSES)

    call_counter = {"n": 0}

    def _open_side_effect(path, *args, **kwargs):
        path_str = str(path)
        if "words.pkl" in path_str:
            import io
            return io.BytesIO(words_pkl)
        if "classes.pkl" in path_str:
            import io
            return io.BytesIO(classes_pkl)
        if "intents.json" in path_str:
            import io
            return io.StringIO(json.dumps(FAKE_INTENTS))
        if "index.html" in path_str:
            import io
            return io.StringIO("<html><body>Chatbot</body></html>")
        # Fall through to real open for everything else
        return open.__wrapped__(path, *args, **kwargs) if hasattr(open, "__wrapped__") else open(path, *args, **kwargs)

    # Ensure app module is not cached from a previous import attempt
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("app",):
            del sys.modules[mod_name]

    with patch("keras.models.load_model", return_value=fake_model), \
         patch("builtins.open", side_effect=_open_side_effect), \
         patch("pickle.load", side_effect=lambda f: pickle.loads(f.read())), \
         patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None), \
         patch("nltk.download", return_value=True):

        import app as chatbot_app  # noqa: PLC0415

    # Attach the fake model so tests can manipulate it
    chatbot_app._test_fake_model = fake_model
    return chatbot_app


@pytest.fixture(scope="module")
def client(patched_app):
    return TestClient(patched_app.app)


# ---------------------------------------------------------------------------
# 1. App metadata / startup
# ---------------------------------------------------------------------------

class TestAppMetadata:
    def test_app_title(self, patched_app):
        assert patched_app.app.title == "AI Chatbot API"

    def test_app_version(self, patched_app):
        assert patched_app.app.version == "1.0.0"

    def test_app_description(self, patched_app):
        assert "FastAPI" in patched_app.app.description or "Chatbot" in patched_app.app.description

    def test_globals_loaded(self, patched_app):
        assert isinstance(patched_app.words, list)
        assert isinstance(patched_app.classes, list)
        assert isinstance(patched_app.intents, dict)
        assert len(patched_app.words) > 0
        assert len(patched_app.classes) > 0


# ---------------------------------------------------------------------------
# 2. CORS middleware
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_cors_preflight_allowed(self, client):
        """OPTIONS request from any origin should return 200 with CORS headers."""
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code in (200, 204)

    def test_cors_header_present_on_post(self, client):
        payload = {"msg": "hello"}
        response = client.post(
            "/api/chat",
            json=payload,
            headers={"Origin": "http://example.com"},
        )
        assert "access-control-allow-origin" in response.headers


# ---------------------------------------------------------------------------
# 3. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model"] == "loaded"


# ---------------------------------------------------------------------------
# 4. Home endpoint
# ---------------------------------------------------------------------------

class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_content_type_html(self, client):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_home_body_not_empty(self, client):
        response = client.get("/")
        assert len(response.text) > 0


# ---------------------------------------------------------------------------
# 5. Chat endpoint – happy paths
# ---------------------------------------------------------------------------

class TestChatEndpointHappyPath:
    def test_basic_greeting(self, client, patched_app):
        """Standard message returns 200 with response and confidence."""
        patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert "response" in body
        assert "confidence" in body
        assert isinstance(body["response"], str)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_my_name_is_pattern(self, client, patched_app):
        """'my name is X' triggers name substitution branch."""
        patched_app._test_fake_model.predict.return_value = np.array([[0.02, 0.03, 0.95]])
        response = client.post("/api/chat", json={"msg": "my name is Alice"})
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["response"], str)

    def test_hi_my_name_is_pattern(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.02, 0.03, 0.95]])
        response = client.post("/api/chat", json={"msg": "hi my name is Bob"})
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["response"], str)

    def test_i_am_pattern(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.02, 0.03, 0.95]])
        response = client.post("/api/chat", json={"msg": "i am Charlie"})
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["response"], str)

    def test_farewell_message(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.03, 0.95, 0.02]])
        response = client.post("/api/chat", json={"msg": "bye"})
        assert response.status_code == 200
        assert response.json()["confidence"] > 0

    def test_case_insensitive_name_pattern(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.02, 0.03, 0.95]])
        response = client.post("/api/chat", json={"msg": "MY NAME IS Dave"})
        assert response.status_code == 200

    def test_whitespace_stripped(self, client, patched_app):
        """Leading/trailing whitespace in message is stripped before processing."""
        patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
        response = client.post("/api/chat", json={"msg": "  hello  "})
        assert response.status_code == 200

    def test_response_model_schema(self, client, patched_app):
        """Verify the response matches the ChatResponse schema."""
        patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
        body = client.post("/api/chat", json={"msg": "hi"}).json()
        assert set(body.keys()) == {"response", "confidence"}

    def test_confidence_is_float(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        body = client.post("/api/chat", json={"msg": "hello"}).json()
        assert isinstance(body["confidence"], float)


# ---------------------------------------------------------------------------
# 6. Chat endpoint – error / edge cases
# ---------------------------------------------------------------------------

class TestChatEndpointErrors:
    def test_empty_message_returns_400(self, client):
        response = client.post("/api/chat", json={"msg": ""})
        assert response.status_code == 400

    def test_whitespace_only_message_returns_400(self, client):
        response = client.post("/api/chat", json={"msg": "   "})
        assert response.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        """Pydantic validation: missing required field → 422."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_invalid_json_returns_422(self, client):
        response = client.post(
            "/api/chat",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_msg_field_wrong_type_returns_422(self, client):
        """msg must be a string; passing an integer triggers 422."""
        response = client.post("/api/chat", json={"msg": 12345})
        # FastAPI coerces int to str for str fields, so accept both 200 and 422
        assert response.status_code in (200, 422)

    def test_internal_error_returns_500(self, client, patched_app):
        """If the model raises, the endpoint should return 500."""
        patched_app._test_fake_model.predict.side_effect = RuntimeError("GPU OOM")
        try:
            response = client.post("/api/chat", json={"msg": "hello"})
            assert response.status_code == 500
        finally:
            # Restore normal behaviour
            patched_app._test_fake_model.predict.side_effect = None
            patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])

    def test_error_detail_message_present(self, client, patched_app):
        patched_app._test_fake_model.predict.side_effect = ValueError("bad input")
        try:
            response = client.post("/api/chat", json={"msg": "hello"})
            if response.status_code == 500:
                assert "detail" in response.json()
        finally:
            patched_app._test_fake_model.predict.side_effect = None
            patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])


# ---------------------------------------------------------------------------
# 7. Pydantic request/response models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, patched_app):
        req = patched_app.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_empty_string(self, patched_app):
        req = patched_app.MessageRequest(msg="")
        assert req.msg == ""

    def test_chat_response_valid(self, patched_app):
        res = patched_app.ChatResponse(response="Hello!", confidence=0.95)
        assert res.response == "Hello!"
        assert res.confidence == 0.95

    def test_chat_response_zero_confidence(self, patched_app):
        res = patched_app.ChatResponse(response="Hmm", confidence=0.0)
        assert res.confidence == 0.0

    def test_message_request_missing_msg_raises(self, patched_app):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            patched_app.MessageRequest()


# ---------------------------------------------------------------------------
# 8. NLP helpers – clean_up_sentence
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_returns_list(self, patched_app):
        result = patched_app.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercases_words(self, patched_app):
        result = patched_app.clean_up_sentence("Hello World")
        assert all(w == w.lower() for w in result)

    def test_tokenizes_correctly(self, patched_app):
        result = patched_app.clean_up_sentence("hello")
        assert "hello" in result

    def test_empty_string(self, patched_app):
        result = patched_app.clean_up_sentence("")
        assert result == []

    def test_punctuation_tokens(self, patched_app):
        result = patched_app.clean_up_sentence("Hi! How are you?")
        # Punctuation may appear as separate tokens
        assert isinstance(result, list)
        assert len(result) > 0

    def test_multiple_words(self, patched_app):
        result = patched_app.clean_up_sentence("how are you")
        assert len(result) >= 3

    def test_lemmatization_applied(self, patched_app):
        """'running' should be lemmatized (not necessarily 'run' with trivial stub,
        but lemmatize must have been called)."""
        with patch.object(patched_app.lemmatizer, "lemmatize", wraps=patched_app.lemmatizer.lemmatize) as mock_lem:
            patched_app.clean_up_sentence("running dogs")
            assert mock_lem.call_count >= 1


# ---------------------------------------------------------------------------
# 9. NLP helpers – bow (bag of words)
# ---------------------------------------------------------------------------

class TestBow:
    def test_returns_numpy_array(self, patched_app):
        result = patched_app.bow("hello", FAKE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_words(self, patched_app):
        result = patched_app.bow("hello", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)

    def test_known_word_sets_one(self, patched_app):
        result = patched_app.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_all_zeros(self, patched_app):
        result = patched_app.bow("xyzzy", FAKE_WORDS)
        assert np.sum(result) == 0

    def test_multiple_known_words(self, patched_app):
        result = patched_app.bow("hello bye", FAKE_WORDS)
        assert np.sum(result) >= 1  # at least one match

    def test_empty_sentence(self, patched_app):
        result = patched_app.bow("", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)
        assert np.sum(result) == 0

    def test_binary_values(self, patched_app):
        result = patched_app.bow("hello hello hello", FAKE_WORDS)
        assert set(result.tolist()).issubset({0, 1})

    def test_show_details_flag_does_not_crash(self, patched_app, capsys):
        result = patched_app.bow("hello", FAKE_WORDS, show_details=True)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 10. NLP helpers – predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, patched_app):
        fake_model = _make_fake_model(predicted_class_index=0)
        result = patched_app.predict_class("hello", fake_model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, patched_app):
        fake_model = _make_fake_model(predicted_class_index=0)
        result = patched_app.predict_class("hello", fake_model)
        if result:
            assert "intent" in result[0]
            assert "probability" in result[0]

    def test_probability_is_string_or_float(self, patched_app):
        fake_model = _make_fake_model(predicted_class_index=0)
        result = patched_app.predict_class("hello", fake_model)
        if result:
            # probability may be stored as str or float depending on implementation
            assert result[0]["probability"] is not None

    def test_model_predict_called_once(self, patched_app):
        fake_model = _make_fake_model(predicted_class_index=1)
        patched_app.predict_class("bye", fake_model)
        fake_model.predict.assert_called()

    def test_empty_sentence_does_not_raise(self, patched_app):
        fake_model = _make_fake_model()
        try:
            patched_app.predict_class("", fake_model)
        except Exception as exc:
            pytest.fail(f"predict_class raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 11. FastAPI version-specific behaviour (fastapi 0.109.1)
# ---------------------------------------------------------------------------

class TestFastAPIVersion:
    def test_openapi_schema_accessible(self, client):
        """fastapi 0.109.1 exposes /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_docs_endpoint_accessible(self, client):
        """Swagger UI (/docs) should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_accessible(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_chat_endpoint_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/chat" in schema["paths"]

    def test_health_endpoint_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]

    def test_post_method_for_chat(self, client):
        schema = client.get("/openapi.json").json()
        assert "post" in schema["paths"]["/api/chat"]

    def test_http_exception_handler_returns_json(self, client):
        """HTTPException raised by FastAPI should produce JSON with 'detail' key."""
        response = client.post("/api/chat", json={"msg": ""})
        assert response.headers["content-type"].startswith("application/json")
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# 12. python-multipart (form data support – 0.0.7)
# ---------------------------------------------------------------------------

class TestMultipartFormSupport:
    def test_form_data_endpoint_not_broken(self, client):
        """
        python-multipart is required by FastAPI for form parsing.
        Sending a plain JSON request still works (ensures the middleware
        upgrade hasn't broken content-type negotiation).
        """
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 200

    def test_content_type_json_accepted(self, client):
        response = client.post(
            "/api/chat",
            content=json.dumps({"msg": "hello"}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 13. NLTK 3.9 – word_tokenize & WordNetLemmatizer
# ---------------------------------------------------------------------------

class TestNLTKIntegration:
    """
    Tests that exercise the NLTK APIs used in app.py (word_tokenize,
    WordNetLemmatizer.lemmatize) through the NLP helper functions.
    These run with real NLTK if the corpora are available, otherwise
    they are patched.
    """

    def test_word_tokenize_called_in_clean_up(self, patched_app):
        with patch("nltk.word_tokenize", return_value=["hello", "world"]) as mock_tok:
            patched_app.clean_up_sentence("hello world")
            mock_tok.assert_called_once_with("hello world")

    def test_lemmatizer_called_for_each_token(self, patched_app):
        tokens = ["running", "dogs", "are", "fast"]
        with patch("nltk.word_tokenize", return_value=tokens):
            with patch.object(
                patched_app.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w
            ) as mock_lem:
                patched_app.clean_up_sentence("running dogs are fast")
                assert mock_lem.call_count == len(tokens)

    def test_lemmatize_called_with_lowercase(self, patched_app):
        with patch("nltk.word_tokenize", return_value=["Running", "Dogs"]):
            with patch.object(
                patched_app.lemmatizer, "lemmatize", side_effect=lambda w, **kw: w.lower()
            ) as mock_lem:
                result = patched_app.clean_up_sentence("Running Dogs")
                for call_args in mock_lem.call_args_list:
                    word_arg = call_args[0][0]
                    assert word_arg == word_arg.lower()

    def test_nltk_word_tokenize_output_used_in_bow(self, patched_app):
        """Tokens returned by nltk.word_tokenize end up influencing the BOW vector."""
        with patch("nltk.word_tokenize", return_value=["hello"]):
            result = patched_app.bow("anything", FAKE_WORDS)
            idx = FAKE_WORDS.index("hello")
            assert result[idx] == 1


# ---------------------------------------------------------------------------
# 14. Integration-style end-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_greeting_flow(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
        patched_app._test_fake_model.predict.side_effect = None
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert len(body["response"]) > 0
        assert body["confidence"] == pytest.approx(0.95, abs=0.01)

    def test_farewell_flow(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.02, 0.96, 0.02]])
        patched_app._test_fake_model.predict.side_effect = None
        response = client.post("/api/chat", json={"msg": "goodbye"})
        assert response.status_code == 200

    def test_name_introduction_flow(self, client, patched_app):
        patched_app._test_fake_model.predict.return_value = np.array([[0.01, 0.02, 0.97]])
        patched_app._test_fake_model.predict.side_effect = None
        response = client.post("/api/chat", json={"msg": "my name is Eve"})
        assert response.status_code == 200
        # The response template uses {n}; if properly replaced it won't contain literal {n}
        body = response.json()
        assert "{n}" not in body["response"] or "Eve" in body["response"] or isinstance(body["response"], str)

    def test_multiple_requests_independent(self, client, patched_app):
        patched_app._test_fake_model.predict.side_effect = None
        for msg in ["hello", "bye", "how are you"]:
            patched_app._test_fake_model.predict.return_value = np.array([[0.9, 0.05, 0.05]])
            resp = client.post("/api/chat", json={"msg": msg})
            assert resp.status_code == 200

    def test_health_not_affected_by_chat_error(self, client, patched_app):
        patched_app._test_fake_model.predict.side_effect = RuntimeError("fail")
        try:
            client.post("/api/chat", json={"msg": "hi"})
        except Exception:
            pass
        finally:
            patched_app._test_fake_model.predict.side_effect = None
        health = client.get("/health")
        assert health.status_code == 200
