import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from birds_dataset import get_dataloaders
from constants import NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_PATH, LABELS
from bird_model import BirdsClassifier


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                num_epochs, final_test_loss=None, final_test_accuracy=None):
    epochs_range = range(1, num_epochs + 1)

    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Loss Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    if final_test_loss is not None:
        plt.scatter([num_epochs], [final_test_loss], color='red', marker='X', s=200, label='Final Test Loss', zorder=5)

    plt.title('Training, Validation, and Final Test Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='major', length=5, width=1.5)
    plt.tight_layout()
    plt.savefig('loss_performance_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Accuracy Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', color='#2ca02c', linewidth=2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='#d62728', linewidth=2)
    if final_test_accuracy is not None:
        plt.scatter([num_epochs], [final_test_accuracy], color='purple', marker='X', s=200, label='Final Test Accuracy', zorder=5)

    plt.title('Training, Validation, and Final Test Accuracy Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='major', length=5, width=1.5)
    plt.tight_layout()
    plt.savefig('accuracy_performance_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis', ax=plt.gca(), xticks_rotation='vertical', values_format='d')
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, linecolor='black', cbar_kws={'label': 'Normalized Count'})
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Normalized Confusion Matrix (Row-wise)', fontsize=18, fontweight='bold')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT      ")
    print("="*50 + "\n")

    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10}")
    print("-" * 90)

    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<25} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1-score']:.4f}     | {metrics['support']:<10}")

    print("-" * 90)
    macro_avg = report.get('macro avg', {})
    weighted_avg = report.get('weighted avg', {})

    print(f"{'Macro Avg':<25} | {macro_avg.get('precision', 0.0):.4f}     | {macro_avg.get('recall', 0.0):.4f}     | {macro_avg.get('f1-score', 0.0):.4f}     | {macro_avg.get('support', 0):<10}")
    print(f"{'Weighted Avg':<25} | {weighted_avg.get('precision', 0.0):.4f}     | {weighted_avg.get('recall', 0.0):.4f}     | {weighted_avg.get('f1-score', 0.0):.4f}     | {weighted_avg.get('support', 0):<10}")
    print("\n" + "="*50 + "\n")


def plot_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    data = []
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            data.append({'Class': class_name, 'Metric': 'Precision', 'Score': metrics['precision']})
            data.append({'Class': class_name, 'Metric': 'Recall', 'Score': metrics['recall']})
            data.append({'Class': class_name, 'Metric': 'F1-Score', 'Score': metrics['f1-score']})

    df = pd.DataFrame(data)

    if df.empty:
        print("No data available to plot classification report. Check if class names match report categories or if test set is empty.")
        return

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Class', y='Score', hue='Metric', data=df, palette='viridis', edgecolor='black')

    plt.title('Classification Report Metrics per Class', fontsize=18, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('classification_report_visual.png', dpi=300, bbox_inches='tight')
    plt.show()


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

    print("\nPlotting training performance curves with final test metrics (separated)...")
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, NUM_EPOCHS,
                           final_test_loss=final_test_loss, final_test_accuracy=final_test_accuracy)

    class_names = LABELS
    print_classification_report(all_labels, all_preds, class_names)

    print("\n--- Visualizing Classification Report ---")
    plot_classification_report(all_labels, all_preds, class_names)

    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(all_labels, all_preds, class_names)
    print("\n--- Normalized Confusion Matrix ---")
    plot_normalized_confusion_matrix(all_labels, all_preds, class_names)

    print("\nMetrics and visualizations generated!")


if __name__ == "__main__":
    train_model()
