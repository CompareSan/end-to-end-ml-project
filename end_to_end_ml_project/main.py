import nltk
import pandas as pd
from nltk.corpus import qc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from end_to_end_ml_project.evaluate import get_accuracy
from end_to_end_ml_project.feature_engineering import tfidf_transform
from end_to_end_ml_project.preprocessing import (
    convert_to_lower_case,
    get_df_without_stop_words,
    remove_punctuation,
    split_labels,
)


def main() -> None:
    nltk.download("qc")
    train_tuples = qc.tuples("train.txt")
    test_tuples = qc.tuples("test.txt")

    train_df = pd.DataFrame(train_tuples, columns=["full_label", "text"])
    test_df = pd.DataFrame(test_tuples, columns=["full_label", "text"])

    train_df = split_labels(train_df)
    test_df = split_labels(test_df)
    train_df = convert_to_lower_case(train_df)
    test_df = convert_to_lower_case(test_df)
    train_df = remove_punctuation(train_df)
    test_df = remove_punctuation(test_df)

    train_df = get_df_without_stop_words(train_df)
    test_df = get_df_without_stop_words(test_df)
    le = LabelEncoder()
    le.fit(train_df["main_cat"])
    train_df["main_cat"] = le.transform(train_df["main_cat"])
    test_df["main_cat"] = le.transform(test_df["main_cat"])

    X = train_df["text"]
    y = train_df["main_cat"]
    X_test = test_df["text"]
    y_test = test_df["main_cat"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    tftfidf_vect = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    X_train_tfidf, X_valid_tfidf, X_test_tfidf = tfidf_transform(
        tftfidf_vect, X_train, X_valid, X_test
    )

    model = LogisticRegression(random_state=0)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    train_acc = get_accuracy(model, X_train_tfidf, y_train)
    valid_acc = get_accuracy(model, X_valid_tfidf, y_valid)
    test_acc = get_accuracy(model, X_test_tfidf, y_test)

    print(f"Accuracy on train: {train_acc}")
    print(f"Accuracy on validation: {valid_acc}")
    print(f"Accuracy on test: {test_acc}")


if __name__ == "__main__":
    main()
