import nltk
import pandas as pd
import torch
from evaluate import evaluate_net
from neural_net import FCNet
from nltk.corpus import qc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tfidf_dataset import TfidfDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from training import train_net

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
    train_dataset = TfidfDataset(X_train_tfidf, y_train)
    valid_dataset = TfidfDataset(X_valid_tfidf, y_valid)
    test_dataset = TfidfDataset(X_test_tfidf, y_test)
    BATCH_SIZE = 64

    tfdif_train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    tfdif_test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    tfdif_valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    LR = 3e-4
    model = FCNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = model.to(device)
    trained_model, _, _ = train_net(
        model,
        optimizer,
        criterion,
        tfdif_train_dataloader,
        tfdif_valid_dataloader,
        device,
        n_epochs=3,
    )

    accuracy, precision, recall, f1 = evaluate_net(
        trained_model,
        tfdif_test_dataloader,
        device,
    )
    print(f"Accuracy: {accuracy}")
    print(f"Precision:, {precision}")
    print(f"Recall:, {recall}")
    print(f"F1 Score:, {f1}")


if __name__ == "__main__":
    main()
