# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import itertools
from sklearn.metrics import confusion_matrix
from skimage.segmentation import mark_boundaries
from data import dataLoaderMaking


def evaluation(model, val_loader, criterion, device, num_classes=5):
    val_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()  # Set the model to evaluation mode
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Remove channel dimension if necessary

        with torch.no_grad():
            output = model(data)

        # Calculate the loss
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Pixel-wise predictions
        _, preds = torch.max(output, dim=1)

        # Calculate correct pixels for each class
        for i in range(num_classes):
            class_mask = (label == i)
            class_correct[i] += torch.sum((preds == i) & class_mask).item()
            class_total[i] += torch.sum(class_mask).item()

    # Calculate the average loss
    val_loss /= len(val_loader.dataset)

    # Display class-wise accuracies
    print('Validation Loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy_class = 100.0 * class_correct[i] / class_total[i]
            print(f'Accuracy for class {i}: {accuracy_class:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Accuracy for class {i}: N/A (no pixels)')

    # Calculate overall accuracy
    total_correct_pixels = sum(class_correct)
    total_pixels = sum(class_total)
    overall_accuracy = 100.0 * total_correct_pixels / total_pixels

    print('Overall pixel-wise accuracy: {:.2f}% ({}/{})'.format(overall_accuracy, total_correct_pixels, total_pixels))

    return val_loss, overall_accuracy, class_correct, class_total

def evaluation_with_dice(model, val_loader, criterion, device, num_classes=5):
    val_loss = 0.0
    dice_scores = [0.0] * num_classes  # Dice scores per class
    class_total = [0] * num_classes

    model.eval()  # Set the model to evaluation mode
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Remove channel dimension if necessary

        with torch.no_grad():
            output = model(data)

        # Calculate the loss
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Pixel-wise predictions
        _, preds = torch.max(output, dim=1)

        # Calculate Dice Score for each class
        for i in range(num_classes):
            class_mask = (label == i)
            pred_mask = (preds == i)

            # Intersection and union for Dice Score
            intersection = torch.sum(pred_mask & class_mask).item()
            union = torch.sum(pred_mask).item() + torch.sum(class_mask).item()

            if union > 0:  # Avoid division by zero
                dice_scores[i] += 2.0 * intersection / union
                class_total[i] += 1

    # Calculate the average loss
    val_loss /= len(val_loader.dataset)

    # Display Dice scores per class
    print('Validation Loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            average_dice = dice_scores[i] / class_total[i]
            print(f'Dice Score for class {i}: {average_dice:.4f}')
        else:
            print(f'Dice Score for class {i}: N/A (no pixels)')

    # Calculate the global average Dice Score
    global_dice = sum(dice_scores) / sum(class_total)
    print('Overall Dice Score: {:.4f}'.format(global_dice))

    return val_loss, global_dice, dice_scores

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def evaluate_confusion_matrix(model, val_loader, device, num_classes=5):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for data, mask in val_loader:
            data = data.to(device)
            mask = mask.squeeze(1).to(device).long()  # Explicit conversion to integers

            # Prédictions du modèle
            output = model(data)
            predicted_classes = torch.argmax(output, dim=1).long()  # Explicit conversion to integers

            # Collecter les valeurs réelles et prédites
            y_true.extend(mask.cpu().numpy().flatten().astype(int))  # Explicit conversion to integers
            y_pred.extend(predicted_classes.cpu().numpy().flatten().astype(int))  # Explicit conversion to integers

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Display the confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=[f'Classe {i}' for i in range(num_classes)])
    plt.show()

def display_random_prediction(model, val_loader, device):
    model.eval() # Set the model to evaluation mode

    # Select a random batch from the validation loader
    data_iter = iter(val_loader)
    random_index = random.randint(0, len(val_loader) - 1)
    
    for _ in range(random_index):
        next(data_iter)  # Skip until the random index

    # Retrieve a batch of data
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Make predictions
    with torch.no_grad():
        prediction = model(data)
        predicted_classes = torch.argmax(prediction, dim=1)

    # Convert to CPU format for display
    data = data.cpu().squeeze(0).squeeze(0).numpy()  # First image of the batch
    mask = mask.cpu().squeeze(0).numpy()  # Ground truth
    predicted_classes = predicted_classes.cpu().squeeze(0).numpy()  # Predictions

    # Normalize the source image for display
    data = (data - data.min()) / (data.max() - data.min())

    # Add contours to the source image
    image_with_contours = mark_boundaries(data, mask, color=(0, 1, 0))  # Green contours (ground truth)
    image_with_contours = mark_boundaries(image_with_contours, predicted_classes, color=(1, 0, 0))  # Red contours (predictions)

    # Display the image with contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('Contours Vrais (Vert) et Prédits (Rouge)')
    plt.axis('off')
    plt.show()

# Faire une fonctionn pour plot les deux models sur un meme masque et pour comparer
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="UNet Model Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for evaluation.")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes in the segmentation task.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument('--image_size', type=int, default=128, help="Size to which images should be resized.")
    parser.add_argument('--criterion', type=str, default="cross_entropy", choices=['cross_entropy'], help="Loss function.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device(args.device)

    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = torch.load(args.model_path, map_location=device)
    model = model.to(device)

    # Define loss function
    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    # Data loading
    print(f"Loading data from {args.data_dir}...")
    # Create dataset and DataLoader
    train_loader,test_loader,val_loader=dataLoaderMaking(namefile=argparse.data_dir,target_shape = (256, 256),batch_size = argparse.batch_size)

    # Perform evaluations
    print("Evaluating model...")
    evaluation(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Evaluating with Dice scores...")
    evaluation_with_dice(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Generating confusion matrix...")
    evaluate_confusion_matrix(model, test_loader, device, num_classes=args.num_classes)

    print("Displaying random predictions...")
    display_random_prediction(model, test_loader, device)
