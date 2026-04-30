"""
Shared pytest fixtures and configuration for the chatbot test suite.
All external I/O (file loading, Keras model, NLTK) is mocked here so that
tests are fully self-contained.
"""
import json
import pickle
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you soon!"],
        },
        {
            "tag": "name",
            "patterns": ["my name is John", "I am Alice"],
            "responses": ["Nice to meet you, {n}!"],
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you"],
            "responses": ["You're welcome!"],
        },
    ]
}

SAMPLE_WORDS = ["are", "bye", "goodbye", "hello", "hey", "hi", "how", "later", "name",
                "see", "thank", "you"]
SAMPLE_CLASSES = ["goodbye", "greeting", "name", "thanks"]


@pytest.fixture
def sample_intents():
    return SAMPLE_INTENTS


@pytest.fixture
def sample_words():
    return SAMPLE_WORDS


@pytest.fixture
def sample_classes():
    return SAMPLE_CLASSES


# ---------------------------------------------------------------------------
# Mock Keras model
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_keras_model():
    """A mock Keras model whose predict() returns a plausible softmax output."""
    model = MagicMock()
    # By default predict returns high confidence for "greeting" (index 1)
    prediction = np.array([[0.05, 0.85, 0.05, 0.05]])
    model.predict.return_value = prediction
    return model


# ---------------------------------------------------------------------------
# App-level fixture: patches all file I/O and model loading before import
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(mock_keras_model, tmp_path):
    """
    Spin up a TestClient for app.py with all heavy dependencies mocked:
      - keras.models.load_model
      - pickle.load  (words.pkl / classes.pkl)
      - builtins.open for intents.json and templates/index.html
      - StaticFiles (directory "static" need not exist)
    """
    from fastapi.testclient import TestClient

    intents_json = json.dumps(SAMPLE_INTENTS)

    def fake_open(file, mode="r", *args, **kwargs):
        if isinstance(file, str) and file.endswith("intents.json"):
            return mock_open(read_data=intents_json)()
        if isinstance(file, str) and "index.html" in file:
            return mock_open(read_data="<html><body>Chatbot</body></html>")()
        # Fallback for pickle.load calls routed through open()
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        m.read.return_value = b""
        return m

    words_bytes = pickle.dumps(SAMPLE_WORDS)
    classes_bytes = pickle.dumps(SAMPLE_CLASSES)
    pickle_call_count = {"n": 0}

    def fake_pickle_load(f):
        n = pickle_call_count["n"]
        pickle_call_count["n"] += 1
        return SAMPLE_WORDS if n == 0 else SAMPLE_CLASSES

    with patch("keras.models.load_model", return_value=mock_keras_model), \
         patch("pickle.load", side_effect=fake_pickle_load), \
         patch("builtins.open", side_effect=fake_open), \
         patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None), \
         patch("fastapi.staticfiles.StaticFiles.__call__", new_callable=MagicMock):

        # Force fresh import
        import importlib, sys
        sys.modules.pop("app", None)
        import app as chatbot_app

        client = TestClient(chatbot_app.app, raise_server_exceptions=False)
        yield client, chatbot_app
