"""
Shared pytest fixtures and configuration for the chatbot test suite.
"""
import json
import pickle
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you"],
            "responses": ["Hello!", "Hi there!", "Hey! How can I help?"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Have a nice day!"],
        },
        {
            "tag": "name",
            "patterns": ["My name is {n}", "I am {n}"],
            "responses": ["Nice to meet you, {n}!", "Hello {n}, how can I help?"],
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Happy to help!"],
        },
    ]
}

SAMPLE_WORDS = [
    "are", "bye", "goodbye", "hello", "help", "hey", "hi",
    "how", "later", "name", "see", "thank", "you",
]

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


@pytest.fixture
def mock_model():
    """A mock Keras model whose predict() returns a plausible probability array."""
    model = MagicMock()
    # Default: highest probability for index 1 (greeting)
    prob_array = np.zeros((1, len(SAMPLE_CLASSES)))
    prob_array[0, 1] = 0.95          # greeting
    prob_array[0, 0] = 0.03          # goodbye
    prob_array[0, 2] = 0.01          # name
    prob_array[0, 3] = 0.01          # thanks
    model.predict.return_value = prob_array
    return model


@pytest.fixture
def intents_file(tmp_path):
    """Write intents.json to a temp directory and return the path."""
    p = tmp_path / "intents.json"
    p.write_text(json.dumps(SAMPLE_INTENTS))
    return p


@pytest.fixture
def words_file(tmp_path):
    p = tmp_path / "words.pkl"
    p.write_bytes(pickle.dumps(SAMPLE_WORDS))
    return p


@pytest.fixture
def classes_file(tmp_path):
    p = tmp_path / "classes.pkl"
    p.write_bytes(pickle.dumps(SAMPLE_CLASSES))
    return p
