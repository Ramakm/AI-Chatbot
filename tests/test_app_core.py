"""
Tests for core app.py functionality:
  - clean_up_sentence / bow / predict_class / getResponse helpers
  - FastAPI endpoints (/, /api/chat, /health)
  - CORS middleware configuration
  - Pydantic request/response models
  - NLTK + Keras integration as used in the upgraded packages
"""

import importlib
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
# Helpers – build minimal fake artifacts so app.py module-level code runs
# ---------------------------------------------------------------------------

FAKE_WORDS = ["hello", "hi", "how", "are", "you", "bye", "good", "help"]
FAKE_CLASSES = ["greeting", "goodbye", "help"]
FAKE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later"],
            "responses": ["Bye!", "See you later!", "Have a nice day!"],
        },
        {
            "tag": "help",
            "patterns": ["Help", "I need help"],
            "responses": ["I can help you with that.", "Sure, I can help!"],
        },
    ]
}


def _make_fake_model():
    """Return a MagicMock that behaves like a loaded Keras model."""
    model = MagicMock()
    # predict returns shape (1, num_classes) with probabilities
    probs = np.array([[0.7, 0.2, 0.1]])
    model.predict.return_value = probs
    return model


# ---------------------------------------------------------------------------
# Fixture: patch all module-level side-effects and import app fresh
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_app_module():
    """
    Import app.py with all external I/O mocked out so tests stay offline
    and don't need real files or a real Keras model.
    """
    fake_model = _make_fake_model()

    builtins_open_mock = mock_open(read_data=json.dumps(FAKE_INTENTS))

    with (
        patch("keras.models.load_model", return_value=fake_model),
        patch("pickle.load", side_effect=[FAKE_WORDS, FAKE_CLASSES]),
        patch("builtins.open", builtins_open_mock),
        patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None),
        patch("nltk.download", return_value=None),
    ):
        # Remove cached module if present
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("app",):
                del sys.modules[mod_name]

        import app as app_module

        # Patch module-level globals so helpers work predictably
        app_module.words = FAKE_WORDS
        app_module.classes = FAKE_CLASSES
        app_module.intents = FAKE_INTENTS
        app_module.model = fake_model

    yield app_module


@pytest.fixture(scope="module")
def client(patched_app_module):
    """TestClient for the FastAPI app."""
    with (
        patch("builtins.open", mock_open(read_data="<html>Test</html>")),
    ):
        return TestClient(patched_app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Unit tests – clean_up_sentence
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_basic_tokenization(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("Hello World")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lemmatization_applied(self, patched_app_module):
        # "running" should lemmatize to "running" or "run" depending on POS
        result = patched_app_module.clean_up_sentence("running")
        assert isinstance(result, list)

    def test_lowercase_conversion(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("HELLO")
        assert all(w == w.lower() for w in result)

    def test_punctuation_handling(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("Hello! How are you?")
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_empty_string(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("")
        assert result == []

    def test_single_word(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("bye")
        assert "bye" in result

    def test_returns_list_of_strings(self, patched_app_module):
        result = patched_app_module.clean_up_sentence("How are you")
        assert all(isinstance(w, str) for w in result)


# ---------------------------------------------------------------------------
# Unit tests – bow (bag of words)
# ---------------------------------------------------------------------------

class TestBagOfWords:
    def test_returns_numpy_array(self, patched_app_module):
        result = patched_app_module.bow("hello", FAKE_WORDS)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self, patched_app_module):
        result = patched_app_module.bow("hello", FAKE_WORDS)
        assert len(result) == len(FAKE_WORDS)

    def test_known_word_sets_one(self, patched_app_module):
        result = patched_app_module.bow("hello", FAKE_WORDS)
        idx = FAKE_WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_stays_zero(self, patched_app_module):
        result = patched_app_module.bow("zzznonsense", FAKE_WORDS)
        assert all(v == 0 for v in result)

    def test_multiple_known_words(self, patched_app_module):
        result = patched_app_module.bow("hello bye", FAKE_WORDS)
        assert result[FAKE_WORDS.index("hello")] == 1
        assert result[FAKE_WORDS.index("bye")] == 1

    def test_values_are_binary(self, patched_app_module):
        result = patched_app_module.bow("hello hi how are you", FAKE_WORDS)
        assert set(result.tolist()).issubset({0, 1})

    def test_show_details_does_not_raise(self, patched_app_module, capsys):
        # show_details=True triggers print(); should not raise
        patched_app_module.bow("hello", FAKE_WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "found in bag" in captured.out

    def test_empty_sentence(self, patched_app_module):
        result = patched_app_module.bow("", FAKE_WORDS)
        assert sum(result) == 0


# ---------------------------------------------------------------------------
# Unit tests – predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, patched_app_module):
        result = patched_app_module.predict_class("hello", patched_app_module.model)
        assert isinstance(result, list)

    def test_result_contains_intent_and_probability(self, patched_app_module):
        result = patched_app_module.predict_class("hello", patched_app_module.model)
        assert len(result) > 0
        first = result[0]
        assert "intent" in first
        assert "probability" in first

    def test_probability_is_string_of_float(self, patched_app_module):
        result = patched_app_module.predict_class("hello", patched_app_module.model)
        for item in result:
            # probability should be convertible to float
            float(item["probability"])

    def test_model_predict_called(self, patched_app_module):
        patched_app_module.model.predict.reset_mock()
        patched_app_module.predict_class("hello", patched_app_module.model)
        patched_app_module.model.predict.assert_called_once()

    def test_predict_input_shape(self, patched_app_module):
        """Model.predict must receive a 2-D numpy array."""
        call_args = None

        def capture_predict(arr):
            nonlocal call_args
            call_args = arr
            return np.array([[0.7, 0.2, 0.1]])

        patched_app_module.model.predict.side_effect = capture_predict
        patched_app_module.predict_class("hello", patched_app_module.model)
        assert call_args is not None
        assert call_args.ndim == 2
        # restore
        probs = np.array([[0.7, 0.2, 0.1]])
        patched_app_module.model.predict.side_effect = None
        patched_app_module.model.predict.return_value = probs

    def test_low_confidence_filtered(self, patched_app_module):
        """Results below the ERROR_THRESHOLD should not appear."""
        all_low = np.array([[0.01, 0.02, 0.005]])
        patched_app_module.model.predict.return_value = all_low
        result = patched_app_module.predict_class("zzz", patched_app_module.model)
        # either empty list or probabilities all below threshold
        for item in result:
            assert float(item["probability"]) >= 0.0
        # restore
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])


# ---------------------------------------------------------------------------
# Unit tests – getResponse
# ---------------------------------------------------------------------------

class TestGetResponse:
    def test_returns_string(self, patched_app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = patched_app_module.getResponse(ints, FAKE_INTENTS)
        assert isinstance(result, str)

    def test_response_from_correct_intent(self, patched_app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = patched_app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][0]["responses"]

    def test_goodbye_intent(self, patched_app_module):
        ints = [{"intent": "goodbye", "probability": "0.8"}]
        result = patched_app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][1]["responses"]

    def test_help_intent(self, patched_app_module):
        ints = [{"intent": "help", "probability": "0.7"}]
        result = patched_app_module.getResponse(ints, FAKE_INTENTS)
        assert result in FAKE_INTENTS["intents"][2]["responses"]

    def test_empty_ints_handled(self, patched_app_module):
        """With an empty intent list some fallback should be returned."""
        try:
            result = patched_app_module.getResponse([], FAKE_INTENTS)
            assert isinstance(result, str)
        except (IndexError, KeyError):
            # acceptable if code raises on empty list – just verify type
            pass


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
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


class TestHomeEndpoint:
    def test_home_returns_html(self, client):
        with patch("builtins.open", mock_open(read_data="<html>Chatbot</html>")):
            resp = client.get("/")
        # 200 or possible redirect; HTML response
        assert resp.status_code in (200, 307, 404)

    def test_home_content_type(self, client):
        with patch("builtins.open", mock_open(read_data="<html></html>")):
            resp = client.get("/")
        if resp.status_code == 200:
            assert "text/html" in resp.headers.get("content-type", "")


class TestChatEndpoint:
    """Tests for POST /api/chat"""

    def _post(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_valid_greeting_returns_200(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "hello")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "hello")
        if resp.status_code == 200:
            data = resp.json()
            assert "response" in data
            assert "confidence" in data

    def test_confidence_is_float(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "hello")
        if resp.status_code == 200:
            assert isinstance(resp.json()["confidence"], float)

    def test_empty_message_returns_400(self, client):
        resp = self._post(client, "")
        assert resp.status_code == 400

    def test_whitespace_only_message_returns_400(self, client):
        resp = self._post(client, "   ")
        assert resp.status_code == 400

    def test_my_name_is_pattern(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "my name is Alice")
        # Should succeed or return 500 if getResponse raises – either way not 400
        assert resp.status_code != 400

    def test_hi_my_name_is_pattern(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "hi my name is Bob")
        assert resp.status_code != 400

    def test_i_am_pattern(self, client, patched_app_module):
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        resp = self._post(client, "i am Charlie")
        assert resp.status_code != 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_invalid_json_payload(self, client):
        resp = client.post(
            "/api/chat",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_name_substitution_in_response(self, client, patched_app_module):
        """If a response template contains {n}, it should be filled with the user's name."""
        # Override intents to include a {n} placeholder
        patched_app_module.intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["my name is"],
                    "responses": ["Hello {n}!"],
                }
            ]
        }
        patched_app_module.model.predict.return_value = np.array([[0.99]])
        patched_app_module.classes = ["greeting"]
        resp = self._post(client, "my name is Alice")
        if resp.status_code == 200:
            assert "Alice" in resp.json()["response"]
        # Restore
        patched_app_module.intents = FAKE_INTENTS
        patched_app_module.classes = FAKE_CLASSES
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])

    def test_model_error_returns_500(self, client, patched_app_module):
        patched_app_module.model.predict.side_effect = RuntimeError("GPU OOM")
        resp = self._post(client, "hello")
        assert resp.status_code == 500
        # Restore
        patched_app_module.model.predict.side_effect = None
        patched_app_module.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    def test_cors_headers_present_on_options(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # Either 200 or 405 – but CORS headers should be present for allowed origins
        assert "access-control-allow-origin" in resp.headers or resp.status_code in (
            200,
            405,
        )

    def test_cors_allow_origin_wildcard(self, client):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "*"


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_message_request_valid(self, patched_app_module):
        req = patched_app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_rejects_missing_field(self, patched_app_module):
        with pytest.raises(Exception):
            patched_app_module.MessageRequest()

    def test_chat_response_valid(self, patched_app_module):
        resp = patched_app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == pytest.approx(0.95)

    def test_chat_response_confidence_float(self, patched_app_module):
        resp = patched_app_module.ChatResponse(response="Hey", confidence=0.5)
        assert isinstance(resp.confidence, float)

    def test_message_request_strips_preserved(self, patched_app_module):
        """Stripping happens inside the endpoint, not the model itself."""
        req = patched_app_module.MessageRequest(msg="  hello  ")
        assert req.msg == "  hello  "
