import string
from typing import Callable, List

import nltk
from nltk.corpus import stopwords


def convert_to_lower_case(text: str) -> str:
    text = text.lower()
    return text


def remove_punctuation(text: str) -> str:
    text = text.replace("[{}]".format(string.punctuation), "")
    return text


def _get_stopwords(language: str = "english") -> List[str]:
    nltk.download("stopwords")
    stop_words = stopwords.words(language)
    remove_list = ["which", "who", "why", "how", "what", "when", "where", "whom"]
    stop_words = [word for word in stop_words if word not in remove_list]
    return stop_words


def _remove_stop_words(text: str, stop_words: List[str]) -> str:
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def get_df_without_stop_words(text: str, fn: Callable = _get_stopwords) -> str:
    stop_words = fn("english")
    text = _remove_stop_words(text, stop_words)
    return text


def preprocessing(text: str) -> str:
    text = convert_to_lower_case(text)
    text = remove_punctuation(text)
    text = get_df_without_stop_words(text)
    return text
