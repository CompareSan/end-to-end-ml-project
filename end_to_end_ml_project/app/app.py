import pickle

import mlflow
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from preprocessing_inference_pipeline import preprocessing
from pydantic import BaseModel

app = FastAPI()


loaded_model = mlflow.pytorch.load_model(
    "../../mlartifacts/261512800025455687/afd6f97f68b14bd89cfd375061b20bf6/artifacts/pytorch_model",
    map_location=torch.device("cpu"),
)


class InputData(BaseModel):
    text: str


@app.post("/predict")
def classify_text(input_data: InputData):
    text = input_data.text
    text = preprocessing(text)
    with open("../../tfidf.pickle", "rb") as handle:
        tfidf_vectorizer = pickle.load(handle)
    tfidf_feature = tfidf_vectorizer.transform([text])
    tfidf_feature = torch.tensor(tfidf_feature.toarray().astype(np.float32).reshape(-1))
    prediction = loaded_model(tfidf_feature)
    prediction_class = torch.argmax(prediction)
    if prediction_class == 0:
        output = "ABBR"
    elif prediction_class == 1:
        output = "DESC"
    elif prediction_class == 2:
        output = "ENTY"
    elif prediction_class == 3:
        output = "HUM"
    elif prediction_class == 4:
        output = "LOC"
    elif prediction_class == 5:
        output = "NUM"

    return {"prediction": output}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
