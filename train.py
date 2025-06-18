import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from birds_dataset import get_dataloaders
from constants import NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_PATH, LABELS
from bird_model import BirdsClassifier




def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"DataLoaders successfully loaded.")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'image_data.csv' is created by running 'data_preparation.py'.")
        return
    except ValueError as e:
        print(e)
        print("Please ensure data directory is not empty.")
        return


    model = BirdsClassifier(num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nStarting training...")
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        print(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        scheduler.step(epoch_val_loss)

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with Val Acc: {best_val_accuracy:.4f} at epoch {epoch+1}")

    print("\nTraining Finished!")

    print("\nEvaluating on Test Set for final metrics and visualizations...")
    model.eval()
    all_preds = []
    all_labels = []

    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing and Collecting Metrics"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    final_test_loss = test_loss / len(test_loader.dataset)
    final_test_accuracy = correct_test / total_test
    print(f"\nFinal Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.4f}")



if __name__ == "__main__":
    train_model()
