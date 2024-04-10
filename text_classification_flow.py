from metaflow import FlowSpec, Parameter, pypi_base, step


@pypi_base(
    python="3.10.12",
    packages={
        "nltk": "3.8.1",
        "pandas": "2.2.1",
        "scikit-learn": "1.4.1.post1",
        "torch": "2.2.2",
        "mlflow": "2.11.3",
    },
)
class TextClassificationFlow(FlowSpec):
    batch_size = Parameter("batch_size", default=64)
    learning_rate = Parameter("learning_rate", default=3e-4)
    epochs = Parameter("epochs", default=5)

    @step
    def start(self):
        import nltk
        import pandas as pd
        from nltk.corpus import qc

        nltk.download("qc")
        train_tuples = qc.tuples("train.txt")
        test_tuples = qc.tuples("test.txt")
        self.train_df = pd.DataFrame(train_tuples, columns=["full_label", "text"])
        self.test_df = pd.DataFrame(test_tuples, columns=["full_label", "text"])
        self.next(self.preprocessing_data)

    @step
    def preprocessing_data(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        from end_to_end_ml_project.preprocessing import preprocessing

        self.train_df = preprocessing(self.train_df)
        self.test_df = preprocessing(self.test_df)

        le = LabelEncoder()
        le.fit(self.train_df["main_cat"])
        self.train_df["main_cat"] = le.transform(self.train_df["main_cat"])
        self.test_df["main_cat"] = le.transform(self.test_df["main_cat"])

        X = self.train_df["text"]
        y = self.train_df["main_cat"]
        X_test = self.test_df["text"]
        y_test = self.test_df["main_cat"]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        self.X_train, self.X_valid, self.y_train, self.y_valid = (
            X_train,
            X_valid,
            y_train,
            y_valid,
        )
        self.X_test, self.y_test = X_test, y_test
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from end_to_end_ml_project.feature_engineering import tfidf_transform
        from end_to_end_ml_project.tfidf_dataset import TfidfDataset

        tftfidf_vect = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
        X_train_tfidf, X_valid_tfidf, X_test_tfidf = tfidf_transform(
            tftfidf_vect, self.X_train, self.X_valid, self.X_test
        )
        self.train_dataset = TfidfDataset(X_train_tfidf, self.y_train)
        self.valid_dataset = TfidfDataset(X_valid_tfidf, self.y_valid)
        self.test_dataset = TfidfDataset(X_test_tfidf, self.y_test)
        self.next(self.train_model)

    @step
    def train_model(self):
        import mlflow
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader

        from end_to_end_ml_project.evaluate import evaluate_net
        from end_to_end_ml_project.neural_net import FCNet
        from end_to_end_ml_project.training import train_net

        mlflow.set_tracking_uri("http://localhost:8089")
        mlflow.set_experiment("Text-Classification")
        tfdif_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        tfdif_test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        tfdif_valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        model = FCNet()
        model = model.to(device)

        with mlflow.start_run():
            params = {
                "batch size": self.batch_size,
                "learning rate": self.learning_rate,
                "epochs": self.epochs,
            }
            mlflow.log_params(params)

            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            trained_model, _, _ = train_net(
                model,
                optimizer,
                criterion,
                tfdif_train_dataloader,
                tfdif_valid_dataloader,
                device,
                n_epochs=self.epochs,
            )

            accuracy, precision, recall, f1 = evaluate_net(
                trained_model,
                tfdif_test_dataloader,
                device,
            )
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(
                trained_model,
                artifact_path="pytorch_text_classification_model",
            )

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.next(self.end)

    @step
    def end(self):
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 Score: {self.f1}")


if __name__ == "__main__":
    TextClassificationFlow()
