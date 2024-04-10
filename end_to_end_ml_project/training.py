import torch


def train_net(
    model,
    optimizer,
    criterion,
    train_dataloader,
    valid_dataloader,
    device,
    n_epochs=3,
):
    len_train_dataloader = len(train_dataloader)
    len_valid_dataloader = len(valid_dataloader)
    train_losses, valid_losses = [], []

    for epoch in range(n_epochs):
        total_train_loss = 0.0
        model.train()
        for data in train_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            total_valid_loss = 0
            for data in valid_dataloader:
                valid_inputs, valid_labels = data
                valid_inputs = valid_inputs.to(device)
                valid_labels = valid_labels.to(device)
                outputs = model(valid_inputs)
                total_valid_loss += criterion(outputs, valid_labels).item()
        train_losses.append(total_train_loss / len_train_dataloader)
        valid_losses.append(total_valid_loss / len_valid_dataloader)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Val Loss: {valid_losses[-1]}"
        )
    print("Finished Training")
    return model, train_losses, valid_losses
