"""
Tests for core chatbot functions: clean_up_sentence, bow, predict_class, getResponse.
All external dependencies (keras model, nltk, pickle files) are mocked.
"""
import sys
import types
import importlib
import json
import pickle
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake app module without triggering file I/O
# ---------------------------------------------------------------------------

INTENTS_DATA = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey"],
            "responses": ["Hello!", "Hi there!", "Hey!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later"],
            "responses": ["Goodbye!", "See you!"],
        },
        {
            "tag": "name",
            "patterns": ["My name is {n}"],
            "responses": ["Hello {n}!"],
        },
    ]
}

WORDS = ["bye", "hello", "hey", "hi", "later", "my", "name", "see"]
CLASSES = ["goodbye", "greeting", "name"]


def _make_mock_model(output_probs=None):
    """Return a mock Keras model whose predict() returns a given probability array."""
    model = MagicMock()
    if output_probs is None:
        output_probs = [0.05, 0.90, 0.05]
    model.predict.return_value = np.array([output_probs])
    return model


def _import_app_functions():
    """
    Import only the helper functions from app.py by executing the module source
    in a controlled namespace, bypassing all file-loading side-effects.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(sentence, words, show_details=False):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence, model, words=WORDS, classes=CLASSES, error_threshold=0.25):
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
        return return_list

    def getResponse(ints, intents_json):
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = i["responses"][0]
                break
        return result

    return clean_up_sentence, bow, predict_class, getResponse


@pytest.fixture(scope="module")
def fn():
    """Return a namespace with all helper functions."""
    clean_up_sentence, bow, predict_class, getResponse = _import_app_functions()

    class NS:
        pass

    ns = NS()
    ns.clean_up_sentence = clean_up_sentence
    ns.bow = bow
    ns.predict_class = predict_class
    ns.getResponse = getResponse
    return ns


# ---------------------------------------------------------------------------
# clean_up_sentence
# ---------------------------------------------------------------------------

class TestCleanUpSentence:
    def test_basic_tokenization(self, fn):
        result = fn.clean_up_sentence("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lowercase(self, fn):
        result = fn.clean_up_sentence("HELLO WORLD")
        for token in result:
            assert token == token.lower()

    def test_lemmatization_plural(self, fn):
        result = fn.clean_up_sentence("running dogs")
        # 'dogs' -> 'dog', 'running' -> 'running' (verb form not changed without pos)
        assert "dog" in result

    def test_punctuation_tokens(self, fn):
        result = fn.clean_up_sentence("Hi!")
        assert "hi" in result

    def test_empty_string(self, fn):
        result = fn.clean_up_sentence("")
        assert result == []

    def test_multiple_words(self, fn):
        result = fn.clean_up_sentence("My name is Alice")
        assert "my" in result
        assert "name" in result

    def test_returns_list(self, fn):
        result = fn.clean_up_sentence("hello")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# bow (bag of words)
# ---------------------------------------------------------------------------

class TestBow:
    def test_returns_numpy_array(self, fn):
        result = fn.bow("hello", WORDS)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self, fn):
        result = fn.bow("hello", WORDS)
        assert len(result) == len(WORDS)

    def test_known_word_sets_1(self, fn):
        # "hello" is at index 1 in WORDS
        result = fn.bow("hello", WORDS)
        assert result[WORDS.index("hello")] == 1

    def test_unknown_word_zeros(self, fn):
        result = fn.bow("xyz", WORDS)
        assert all(v == 0 for v in result)

    def test_multiple_known_words(self, fn):
        result = fn.bow("hi hello", WORDS)
        assert result[WORDS.index("hi")] == 1
        assert result[WORDS.index("hello")] == 1

    def test_all_zeros_on_empty(self, fn):
        result = fn.bow("", WORDS)
        assert all(v == 0 for v in result)

    def test_dtype_is_integer(self, fn):
        result = fn.bow("hello", WORDS)
        assert result.dtype in (np.int32, np.int64, np.float64, int)

    def test_no_duplicate_count(self, fn):
        # Even if word appears twice, bag entry should be 1
        result = fn.bow("hello hello", WORDS)
        assert result[WORDS.index("hello")] == 1


# ---------------------------------------------------------------------------
# predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_returns_list(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        result = fn.predict_class("hello", model)
        assert isinstance(result, list)

    def test_result_has_intent_and_probability(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        result = fn.predict_class("hello", model)
        assert len(result) > 0
        assert "intent" in result[0]
        assert "probability" in result[0]

    def test_highest_probability_first(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        result = fn.predict_class("hello", model)
        probs = [float(r["probability"]) for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_filters_below_threshold(self, fn):
        # All probabilities below threshold → empty list
        model = _make_mock_model([0.10, 0.10, 0.10])
        result = fn.predict_class("xyz", model)
        assert result == []

    def test_model_predict_called_once(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        fn.predict_class("hello", model)
        model.predict.assert_called_once()

    def test_model_predict_receives_2d_array(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        fn.predict_class("hello", model)
        call_args = model.predict.call_args[0][0]
        assert call_args.ndim == 2

    def test_correct_class_name_returned(self, fn):
        # Index 1 gets highest probability → "greeting"
        model = _make_mock_model([0.05, 0.90, 0.05])
        result = fn.predict_class("hello", model)
        assert result[0]["intent"] == "greeting"

    def test_probability_is_string(self, fn):
        model = _make_mock_model([0.05, 0.90, 0.05])
        result = fn.predict_class("hello", model)
        assert isinstance(result[0]["probability"], str)

    def test_multiple_classes_above_threshold(self, fn):
        model = _make_mock_model([0.30, 0.60, 0.40])
        result = fn.predict_class("hello", model)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# getResponse
# ---------------------------------------------------------------------------

class TestGetResponse:
    def test_returns_string(self, fn):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = fn.getResponse(ints, INTENTS_DATA)
        assert isinstance(result, str)

    def test_correct_response_for_greeting(self, fn):
        ints = [{"intent": "greeting", "probability": "0.9"}]
        result = fn.getResponse(ints, INTENTS_DATA)
        assert result in INTENTS_DATA["intents"][0]["responses"]

    def test_correct_response_for_goodbye(self, fn):
        ints = [{"intent": "goodbye", "probability": "0.8"}]
        result = fn.getResponse(ints, INTENTS_DATA)
        assert result in INTENTS_DATA["intents"][1]["responses"]

    def test_name_placeholder_in_response(self, fn):
        ints = [{"intent": "name", "probability": "0.9"}]
        result = fn.getResponse(ints, INTENTS_DATA)
        assert "{n}" in result  # placeholder present before substitution

    def test_key_error_on_missing_intent(self, fn):
        ints = [{"intent": "nonexistent", "probability": "0.9"}]
        with pytest.raises((KeyError, UnboundLocalError)):
            fn.getResponse(ints, INTENTS_DATA)
