import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_net(
    model,
    dataloader,
    device,
):
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
            true_labels += labels.tolist()
    accuracy = correct / total

    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    f1 = f1_score(true_labels, predictions, average="macro")
    return accuracy, precision, recall, f1
