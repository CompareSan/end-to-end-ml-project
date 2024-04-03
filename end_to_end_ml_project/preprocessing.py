import string
from typing import Callable, List

import nltk
import pandas as pd
from nltk.corpus import stopwords


def split_labels(df: pd.DataFrame) -> pd.DataFrame:
    df[["main_cat", "gran_cat"]] = df["full_label"].str.split(":", expand=True)
    return df


def convert_to_lower_case(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].str.lower()
    return df


def remove_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].str.replace("[{}]".format(string.punctuation), "")
    return df


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


def get_df_without_stop_words(
    df: pd.DataFrame, fn: Callable = _get_stopwords
) -> pd.DataFrame:
    stop_words = fn("english")
    df["text"] = df["text"].apply(lambda x: _remove_stop_words(x, stop_words))
    return df
