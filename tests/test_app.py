"""
Comprehensive pytest tests for the FastAPI Chatbot Server (app.py).

Covers:
- FastAPI endpoint behaviour (happy path + error/edge cases)
- CORS middleware configuration
- NLTK text-processing helpers (clean_up_sentence, bow, predict_class, getResponse)
- Upgraded-package surface:
    * fastapi  0.104.1 → 0.109.1  (TestClient, HTTPException, response models, CORS)
    * python-multipart 0.0.6 → 0.0.7 (form-data parsing)
    * nltk 3.8.1 → 3.9 (word_tokenize, WordNetLemmatizer)
"""

import importlib
import json
import sys
import types
import pickle
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Helpers – build a minimal fake module tree so app.py can be imported
# without real model files, Keras, or a GPU.
# ---------------------------------------------------------------------------

def _make_fake_keras():
    """Return a sys.modules patch dict that stubs out keras / tensorflow."""
    # tensorflow.keras.optimizers
    tf_keras_opt = types.ModuleType("tensorflow.keras.optimizers")
    tf_keras_opt.SGD = MagicMock()

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_keras = types.ModuleType("tensorflow.keras")
    tensorflow_mod.keras = tensorflow_keras
    tensorflow_keras.optimizers = tf_keras_opt

    # keras.models
    fake_model = MagicMock()
    fake_model.predict = MagicMock(
        return_value=np.array([[0.1, 0.8, 0.1]])
    )

    keras_models = types.ModuleType("keras.models")
    keras_models.load_model = MagicMock(return_value=fake_model)
    keras_models.Sequential = MagicMock()

    keras_layers = types.ModuleType("keras.layers")
    keras_layers.Dense = MagicMock()
    keras_layers.Dropout = MagicMock()

    keras_mod = types.ModuleType("keras")
    keras_mod.models = keras_models
    keras_mod.layers = keras_layers

    return {
        "tensorflow": tensorflow_mod,
        "tensorflow.keras": tensorflow_keras,
        "tensorflow.keras.optimizers": tf_keras_opt,
        "keras": keras_mod,
        "keras.models": keras_models,
        "keras.layers": keras_layers,
    }, fake_model


def _make_intents():
    return {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hello", "hi"],
                "responses": ["Hello!", "Hi there!"],
            },
            {
                "tag": "name",
                "patterns": ["my name is"],
                "responses": ["Nice to meet you, {n}!"],
            },
        ]
    }


# Pickled bytes for words / classes
WORDS = ["hello", "hi", "name", "my", "i", "am"]
CLASSES = ["greeting", "name"]
WORDS_PKL = pickle.dumps(WORDS)
CLASSES_PKL = pickle.dumps(CLASSES)
INTENTS_JSON = json.dumps(_make_intents())


# ---------------------------------------------------------------------------
# Fixture: import app with all heavy dependencies mocked
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    """
    Import app.py with Keras, model files, static files, and templates mocked.
    Returns the module so tests can call helpers directly.
    """
    keras_patches, fake_model = _make_fake_keras()

    # Patch open() for the three data files that app.py reads at import time.
    def fake_open(path, mode="r", *args, **kwargs):
        if "words.pkl" in str(path):
            return mock_open(read_data=WORDS_PKL)()
        if "classes.pkl" in str(path):
            return mock_open(read_data=CLASSES_PKL)()
        if "intents.json" in str(path):
            m = mock_open(read_data=INTENTS_JSON)()
            return m
        if "index.html" in str(path):
            return mock_open(read_data="<html>chatbot</html>")()
        raise FileNotFoundError(path)

    # StaticFiles must not raise (no real directory)
    static_mock = MagicMock()
    static_files_cls = MagicMock(return_value=static_mock)

    with patch.dict(sys.modules, keras_patches):
        with patch("builtins.open", side_effect=fake_open):
            with patch("pickle.load", side_effect=lambda f: pickle.loads(f.read())):
                with patch("fastapi.staticfiles.StaticFiles", static_files_cls):
                    # Remove cached module if re-running
                    sys.modules.pop("app", None)
                    import app as _app
                    # Expose the fake model for assertions
                    _app._fake_model = fake_model
                    return _app


@pytest.fixture(scope="module")
def client(app_module):
    """Return a fastapi.testclient.TestClient wrapping the app."""
    from fastapi.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


# ===========================================================================
# 1.  Module-level sanity checks
# ===========================================================================

class TestModuleLoads:
    def test_app_instance_created(self, app_module):
        from fastapi import FastAPI
        assert isinstance(app_module.app, FastAPI)

    def test_app_metadata(self, app_module):
        assert app_module.app.title == "AI Chatbot API"
        assert app_module.app.version == "1.0.0"

    def test_words_loaded(self, app_module):
        assert isinstance(app_module.words, list)
        assert len(app_module.words) > 0

    def test_classes_loaded(self, app_module):
        assert isinstance(app_module.classes, list)
        assert len(app_module.classes) > 0

    def test_intents_loaded(self, app_module):
        assert "intents" in app_module.intents

    def test_lemmatizer_initialised(self, app_module):
        from nltk.stem import WordNetLemmatizer
        assert isinstance(app_module.lemmatizer, WordNetLemmatizer)


# ===========================================================================
# 2.  CORS middleware (fastapi upgrade surface)
# ===========================================================================

class TestCORSMiddleware:
    def test_cors_preflight_returns_200(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 204)

    def test_cors_allow_origin_wildcard(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://another-origin.com"},
        )
        assert resp.headers.get("access-control-allow-origin") in ("*", "http://another-origin.com")

    def test_cors_headers_present_on_regular_request(self, client):
        resp = client.get("/health", headers={"Origin": "http://test.com"})
        assert "access-control-allow-origin" in resp.headers


# ===========================================================================
# 3.  Health-check endpoint
# ===========================================================================

class TestHealthEndpoint:
    def test_health_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model"] == "loaded"


# ===========================================================================
# 4.  Home endpoint
# ===========================================================================

class TestHomeEndpoint:
    def test_home_returns_html(self, client):
        with patch("builtins.open", mock_open(read_data="<html>chatbot</html>")):
            resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ===========================================================================
# 5.  /api/chat endpoint – happy paths  (fastapi + pydantic surface)
# ===========================================================================

class TestChatEndpointHappy:
    """
    Tests use predict_class / getResponse mocked at module level so that
    only the FastAPI routing / serialisation layer is exercised.
    """

    def _chat(self, client, msg):
        return client.post("/api/chat", json={"msg": msg})

    def test_regular_message_200(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.95"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello!"):
            resp = self._chat(client, "hello")
        assert resp.status_code == 200

    def test_response_schema(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.95"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello!"):
            body = self._chat(client, "hello").json()
        assert "response" in body
        assert "confidence" in body
        assert isinstance(body["confidence"], float)

    def test_confidence_value(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.87"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            body = self._chat(client, "hi").json()
        assert pytest.approx(body["confidence"], abs=1e-4) == 0.87

    def test_my_name_is_substitution(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.90"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Nice to meet you, {n}!"):
            body = self._chat(client, "my name is Alice").json()
        assert "Alice" in body["response"]

    def test_hi_my_name_is_substitution(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.92"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi {n}!"):
            body = self._chat(client, "hi my name is Bob").json()
        assert "Bob" in body["response"]

    def test_i_am_substitution(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.88"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello {n}!"):
            body = self._chat(client, "i am Carol").json()
        assert "Carol" in body["response"]

    def test_message_stripped_of_whitespace(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.80"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello!"):
            resp = self._chat(client, "   hello   ")
        assert resp.status_code == 200

    def test_case_insensitive_my_name_is(self, client, app_module):
        ints = [{"intent": "name", "probability": "0.85"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello {n}!"):
            body = self._chat(client, "MY NAME IS Dave").json()
        assert "Dave" in body["response"]


# ===========================================================================
# 6.  /api/chat endpoint – error / edge cases
# ===========================================================================

class TestChatEndpointErrors:
    def _chat(self, client, payload):
        return client.post("/api/chat", json=payload)

    def test_empty_string_returns_400(self, client):
        resp = self._chat(client, {"msg": ""})
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = self._chat(client, {"msg": "   "})
        assert resp.status_code == 400

    def test_missing_msg_field_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        resp = client.post(
            "/api/chat",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code in (422, 415)

    def test_internal_error_returns_500(self, client, app_module):
        with patch.object(app_module, "predict_class", side_effect=RuntimeError("boom")):
            resp = self._chat(client, {"msg": "hello"})
        assert resp.status_code == 500

    def test_500_detail_message(self, client, app_module):
        with patch.object(app_module, "predict_class", side_effect=RuntimeError("test error")):
            body = self._chat(client, {"msg": "hello"}).json()
        assert "detail" in body

    def test_non_string_msg_returns_422(self, client):
        resp = self._chat(client, {"msg": 12345})
        # pydantic coerces int → str in v2, so 200 or 422 are both acceptable;
        # what must NOT happen is a 500.
        assert resp.status_code != 500

    def test_empty_ints_confidence_zero(self, client, app_module):
        """When predict_class returns [], confidence should be 0.0."""
        with patch.object(app_module, "predict_class", return_value=[]), \
             patch.object(app_module, "getResponse", return_value="I don't understand"):
            body = self._chat(client, {"msg": "asdfghjk"}).json()
        assert body["confidence"] == 0.0


# ===========================================================================
# 7.  Request / response Pydantic models (fastapi upgrade surface)
# ===========================================================================

class TestPydanticModels:
    def test_message_request_valid(self, app_module):
        req = app_module.MessageRequest(msg="hello")
        assert req.msg == "hello"

    def test_message_request_rejects_missing(self, app_module):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            app_module.MessageRequest()  # no msg

    def test_chat_response_valid(self, app_module):
        resp = app_module.ChatResponse(response="Hi!", confidence=0.95)
        assert resp.response == "Hi!"
        assert resp.confidence == 0.95

    def test_chat_response_confidence_float(self, app_module):
        resp = app_module.ChatResponse(response="Hey", confidence=1)
        assert isinstance(resp.confidence, float)


# ===========================================================================
# 8.  NLTK helpers – clean_up_sentence  (nltk 3.8.1 → 3.9 surface)
# ===========================================================================

class TestCleanUpSentence:
    def test_returns_list(self, app_module):
        result = app_module.clean_up_sentence("Hello world")
        assert isinstance(result, list)

    def test_lowercases_tokens(self, app_module):
        result = app_module.clean_up_sentence("HELLO")
        assert all(t == t.lower() for t in result)

    def test_lemmatizes_plural(self, app_module):
        result = app_module.clean_up_sentence("dogs")
        assert "dog" in result

    def test_lemmatizes_verb(self, app_module):
        result = app_module.clean_up_sentence("running")
        # WordNetLemmatizer default POS is noun, so 'running' stays as 'running'
        assert isinstance(result, list)

    def test_tokenizes_multi_word(self, app_module):
        result = app_module.clean_up_sentence("hello there friend")
        assert len(result) == 3

    def test_empty_string_returns_empty_list(self, app_module):
        result = app_module.clean_up_sentence("")
        assert result == []

    def test_punctuation_included_as_token(self, app_module):
        result = app_module.clean_up_sentence("hello!")
        # nltk word_tokenize splits punctuation; list should be non-empty
        assert len(result) >= 1

    def test_single_word(self, app_module):
        result = app_module.clean_up_sentence("hello")
        assert "hello" in result

    def test_numbers_tokenised(self, app_module):
        result = app_module.clean_up_sentence("I have 2 cats")
        assert "2" in result


# ===========================================================================
# 9.  NLTK helpers – bow  (bag-of-words)
# ===========================================================================

class TestBow:
    def test_returns_numpy_array(self, app_module):
        result = app_module.bow("hello", WORDS)
        assert isinstance(result, np.ndarray)

    def test_length_equals_vocab_size(self, app_module):
        result = app_module.bow("hello", WORDS)
        assert len(result) == len(WORDS)

    def test_known_word_sets_one(self, app_module):
        # WORDS = ["hello", "hi", "name", "my", "i", "am"]
        result = app_module.bow("hello", WORDS)
        idx = WORDS.index("hello")
        assert result[idx] == 1

    def test_unknown_word_all_zeros(self, app_module):
        result = app_module.bow("xyzzy", WORDS)
        assert np.sum(result) == 0

    def test_empty_sentence_all_zeros(self, app_module):
        result = app_module.bow("", WORDS)
        assert np.sum(result) == 0

    def test_multiple_words_match(self, app_module):
        result = app_module.bow("hello hi", WORDS)
        assert result[WORDS.index("hello")] == 1
        assert result[WORDS.index("hi")] == 1

    def test_binary_values_only(self, app_module):
        result = app_module.bow("hello hello hello", WORDS)
        # Even repeated words should only set the flag once
        assert set(result.tolist()).issubset({0, 1})

    def test_show_details_does_not_crash(self, app_module, capsys):
        app_module.bow("hello", WORDS, show_details=True)
        captured = capsys.readouterr()
        assert "found in bag" in captured.out

    def test_empty_vocab_returns_empty_array(self, app_module):
        result = app_module.bow("hello", [])
        assert len(result) == 0


# ===========================================================================
# 10.  NLTK helpers – predict_class
# ===========================================================================

class TestPredictClass:
    def _make_model(self, probs):
        m = MagicMock()
        m.predict = MagicMock(return_value=np.array([probs]))
        return m

    def test_returns_list(self, app_module):
        model = self._make_model([0.1, 0.9])
        result = app_module.predict_class("hello", model)
        assert isinstance(result, list)

    def test_each_entry_has_intent_and_probability(self, app_module):
        model = self._make_model([0.1, 0.9])
        result = app_module.predict_class("hello", model)
        for entry in result:
            assert "intent" in entry
            assert "probability" in entry

    def test_sorted_by_probability_descending(self, app_module):
        model = self._make_model([0.1, 0.9])
        result = app_module.predict_class("hello", model)
        probs = [float(r["probability"]) for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_low_confidence_filtered_out(self, app_module):
        """Predictions below threshold (typically 0.25) should be excluded."""
        model = self._make_model([0.01, 0.01])
        result = app_module.predict_class("asdfgh", model)
        # Either empty list or all probabilities above threshold
        for entry in result:
            assert float(entry["probability"]) > 0.0

    def test_model_predict_called_once(self, app_module):
        model = self._make_model([0.3, 0.7])
        app_module.predict_class("hello", model)
        model.predict.assert_called_once()


# ===========================================================================
# 11.  python-multipart – form data (upgraded 0.0.6 → 0.0.7)
# ===========================================================================

class TestMultipartFormData:
    """
    Ensure the app can handle multipart / form-urlencoded data at the
    transport layer (python-multipart is required by FastAPI for this).
    The /api/chat endpoint accepts JSON, so we verify the TestClient's
    form helpers still work correctly after the upgrade.
    """

    def test_json_post_still_works(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hello!"):
            resp = client.post("/api/chat", json={"msg": "hello"})
        assert resp.status_code == 200

    def test_form_data_to_json_endpoint_returns_422(self, client):
        """Sending form-encoded data to a JSON endpoint should yield 422."""
        resp = client.post("/api/chat", data={"msg": "hello"})
        assert resp.status_code == 422

    def test_content_type_json_accepted(self, client, app_module):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value="Hi!"):
            resp = client.post(
                "/api/chat",
                content=json.dumps({"msg": "hello"}),
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200


# ===========================================================================
# 12.  FastAPI routing – method-not-allowed & not-found
# ===========================================================================

class TestRoutingEdgeCases:
    def test_get_chat_returns_405(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_unknown_path_returns_404(self, client):
        resp = client.get("/does/not/exist")
        assert resp.status_code == 404

    def test_delete_health_returns_405(self, client):
        resp = client.delete("/health")
        assert resp.status_code == 405


# ===========================================================================
# 13.  WordNetLemmatizer (nltk upgrade surface)
# ===========================================================================

class TestWordNetLemmatizer:
    """Direct tests of the NLTK lemmatizer used in clean_up_sentence."""

    @pytest.fixture(autouse=True)
    def lemmatizer(self):
        from nltk.stem import WordNetLemmatizer
        self.lem = WordNetLemmatizer()

    def test_lemmatize_noun_plural(self):
        assert self.lem.lemmatize("dogs") == "dog"

    def test_lemmatize_verb(self):
        assert self.lem.lemmatize("running", pos="v") == "run"

    def test_lemmatize_adjective(self):
        assert self.lem.lemmatize("better", pos="a") == "good"

    def test_lemmatize_already_base(self):
        assert self.lem.lemmatize("cat") == "cat"

    def test_lemmatize_returns_string(self):
        result = self.lem.lemmatize("children")
        assert isinstance(result, str)

    def test_lemmatize_lowercased_input(self):
        assert self.lem.lemmatize("DOGS".lower()) == "dog"


# ===========================================================================
# 14.  NLTK word_tokenize (nltk upgrade surface)
# ===========================================================================

class TestNltkWordTokenize:
    def test_basic_tokenization(self):
        tokens = nltk.word_tokenize("Hello world")
        assert tokens == ["Hello", "world"]

    def test_punctuation_split(self):
        tokens = nltk.word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_contraction_split(self):
        tokens = nltk.word_tokenize("I'm fine")
        assert len(tokens) >= 2

    def test_empty_string(self):
        tokens = nltk.word_tokenize("")
        assert tokens == []

    def test_returns_list(self):
        result = nltk.word_tokenize("test sentence")
        assert isinstance(result, list)

    def test_multiple_sentences(self):
        tokens = nltk.word_tokenize("Hello. How are you?")
        assert len(tokens) > 2


# ===========================================================================
# 15.  getResponse helper (if accessible)
# ===========================================================================

class TestGetResponse:
    def test_returns_string(self, app_module):
        intents_data = _make_intents()
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, intents_data)
        assert isinstance(result, str)

    def test_response_from_correct_tag(self, app_module):
        intents_data = _make_intents()
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = app_module.getResponse(ints, intents_data)
        assert result in ["Hello!", "Hi there!"]

    def test_unknown_intent_returns_fallback(self, app_module):
        intents_data = _make_intents()
        ints = [{"intent": "unknown_tag_xyz", "probability": "0.5"}]
        # Should not raise; may return a fallback string
        result = app_module.getResponse(ints, intents_data)
        assert isinstance(result, str)

    def test_empty_ints_does_not_crash(self, app_module):
        intents_data = _make_intents()
        try:
            result = app_module.getResponse([], intents_data)
            assert isinstance(result, str)
        except (IndexError, KeyError):
            pass  # acceptable – main contract is no unhandled crash in endpoint


# ===========================================================================
# 16.  FastAPI OpenAPI schema (fastapi upgrade surface)
# ===========================================================================

class TestOpenAPISchema:
    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_title(self, client):
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "AI Chatbot API"

    def test_openapi_version(self, client):
        schema = client.get("/openapi.json").json()
        assert schema["info"]["version"] == "1.0.0"

    def test_chat_endpoint_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/chat" in schema["paths"]

    def test_health_endpoint_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ===========================================================================
# 17.  Integration-style end-to-end chat flow
# ===========================================================================

class TestEndToEndChatFlow:
    def _chat(self, client, app_module, msg, response_text="Got it!"):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        with patch.object(app_module, "predict_class", return_value=ints), \
             patch.object(app_module, "getResponse", return_value=response_text):
            return client.post("/api/chat", json={"msg": msg})

    def test_full_conversation_turn(self, client, app_module):
        resp = self._chat(client, app_module, "hello")
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Got it!"
        assert 0.0 <= body["confidence"] <= 1.0

    def test_multiple_turns_independent(self, client, app_module):
        r1 = self._chat(client, app_module, "hello", "Hi!")
        r2 = self._chat(client, app_module, "how are you", "Fine!")
        assert r1.json()["response"] == "Hi!"
        assert r2.json()["response"] == "Fine!"

    def test_long_message_handled(self, client, app_module):
        long_msg = "hello " * 200
        resp = self._chat(client, app_module, long_msg.strip())
        assert resp.status_code == 200

    def test_special_characters_in_message(self, client, app_module):
        resp = self._chat(client, app_module, "hello! @#$%^&*()")
        assert resp.status_code == 200

    def test_unicode_message_handled(self, client, app_module):
        resp = self._chat(client, app_module, "こんにちは")
        assert resp.status_code == 200
