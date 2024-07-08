import torch
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from model import TransformerClassifier, get_model
from data import get_training_data
from export import export_onnx

def train(model, dataset, labels, device):

    batch_size = 64

    dl = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=dataset.collate_fn, 
            pin_memory=True
    )

    class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels), 
            y=labels.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    step_size = 3
    num_epochs = 6 * step_size # Number of training epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(dl, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for input, label in progress_bar:
            input = input.to(device)
            label = label.to(device)

            # Forward pass, backward pass, and optimize
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Calculate loss and accuracy
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == label).sum().item()
            total_predictions += label.size(0)

            # Update progress bar description
            avg_loss = total_loss / total_predictions
            accuracy = correct_predictions / total_predictions * 100
            progress_bar.set_postfix(loss=avg_loss, accuracy=f'{accuracy:.2f}%')

        # Print average loss and accuracy for this epoch
        avg_loss = total_loss / len(dl)
        accuracy = correct_predictions / total_predictions * 100
        print(f'End of Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    model.eval()

    torch.save(model.state_dict(), "classifier.pth")

    return model

if __name__ == "__main__":

    model, vocab, device = get_model()

    labels, dataset = get_training_data("training_data.csv", vocab)

    model = train(model, dataset, labels, device)
