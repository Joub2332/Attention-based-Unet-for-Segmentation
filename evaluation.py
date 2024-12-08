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
from model import UNet2d
from model import UNetAug2D


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
    print("test")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    print("test")
    # Display the confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=[f'Classe {i}' for i in range(num_classes)])
    plt.show()

def display_random_prediction(model, val_loader, device):
    # Mettre le modèle en mode évaluation
    model.eval()

    # Sélectionner un batch aléatoire du valid loader
    data_iter = iter(val_loader)
    random_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_index):
        next(data_iter)  # Ignorer jusqu'à l'index aléatoire

    # Récupérer un batch de données
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Faire des prédictions
    with torch.no_grad():
        prediction = model(data)
        predicted_classes = torch.argmax(prediction, dim=1)

    # Choisir une image aléatoire dans le batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()
    predicted_image = predicted_classes[random_image_index].cpu().numpy()

    # Normalisation de l'image source pour affichage
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Ajouter les contours à l'image source
    image_with_contours = mark_boundaries(data_image, mask_image, color=(0, 1, 0))  # Contours verts (vérité terrain)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image, color=(1, 0, 0))  # Contours rouges (prédictions)

    # Afficher l'image avec les contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('Contours Vrais (Vert) et Prédits (Rouge)')
    plt.axis('off')
    plt.show()

def display_comparison(models, val_loader, device, class_names=None):
    """
    Displays a random image with the ground truth mask and predictions from two models.

    Args:
        models: List of trained models [model_1, model_2].
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        class_names: List of class names corresponding to the indices, optional.
    """
    # Set both models to evaluation mode
    for model in models:
        model.eval()

    # Select a random batch from the validation DataLoader
    data_iter = iter(val_loader)
    random_batch_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_batch_index):
        next(data_iter)  # Skip to the selected random batch

    # Get a batch of data and its masks
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Select a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Get predictions from both models
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(data)
            predicted_classes = torch.argmax(prediction, dim=1)
            predictions.append(predicted_classes[random_image_index].cpu().numpy())

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Plot ground truth and predictions
    plt.figure(figsize=(15, 5))

    # Plot the input image
    plt.subplot(1, len(models) + 2, 1)
    plt.imshow(data_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Plot the ground truth mask
    plt.subplot(1, len(models) + 2, 2)
    plt.imshow(mask_image, cmap='viridis')
    if class_names:
        plt.title('Ground Truth')
    else:
        plt.title('Ground Truth Mask')
    plt.axis('off')

    # Plot predictions for each model
    for i, prediction in enumerate(predictions):
        plt.subplot(1, len(models) + 2, i + 3)
        plt.imshow(prediction, cmap='viridis')
        title = f'Prediction (Model {i + 1})'
        if class_names:
            title += f'\n{class_names}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_prediction_for_class(model, val_loader, device, target_class):
    """
    Displays a prediction for a specific class, highlighting true and predicted contours,
    and shows the full mask for the target class.
    
    Args:
        model: Trained segmentation model.
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        target_class: Index of the target class to display (e.g., 1 for liver).
    """
    # Set the model to evaluation mode
    model.eval()

    # Iterate through the validation DataLoader to find a batch with the target class
    for data, mask in val_loader:
        data = data.to(device, dtype=torch.float32)
        mask = mask.squeeze(1).to(device, dtype=torch.long)

        # Check if any mask contains the target class
        contains_target_class = (mask == target_class).any(dim=(1, 2))  # Boolean array per image in batch
        if contains_target_class.any():  # If at least one image contains the target class
            break
    else:
        print(f"No images with class {target_class} were found in the validation set.")
        return

    # Select a random image that contains the target class
    indices_with_target_class = torch.where(contains_target_class)[0]
    random_image_index = random.choice(indices_with_target_class)  # Randomly select one
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Make predictions
    with torch.no_grad():
        prediction = model(data)
        predicted_classes = torch.argmax(prediction, dim=1)
        predicted_image = predicted_classes[random_image_index].cpu().numpy()

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Create a smoothed mask for contours to avoid dotted lines (more continuous boundaries)
    smoothed_mask = np.pad(mask_image == target_class, pad_width=1, mode='constant', constant_values=False)
    smoothed_mask = smoothed_mask[1:-1, 1:-1]  # Remove padding for final mask

    smoothed_pred = np.pad(predicted_image == target_class, pad_width=1, mode='constant', constant_values=False)
    smoothed_pred = smoothed_pred[1:-1, 1:-1]  # Remove padding for final predicted mask

    # Add contours for the target class (Ground truth and Prediction)
    image_with_contours = mark_boundaries(data_image, smoothed_mask, color=(0, 1, 0), mode='thick')  # Green for ground truth
    image_with_contours = mark_boundaries(image_with_contours, smoothed_pred, color=(1, 0, 0), mode='thick')  # Red for predictions

    # Create a full mask for the target class
    full_mask = mask_image == target_class
    full_pred_mask = predicted_image == target_class

    # Display the images
    plt.figure(figsize=(12, 12))

    # Plot the original image with contours
    plt.subplot(2, 2, 1)
    plt.imshow(image_with_contours)
    plt.title(f'Class {target_class}: True contours (green) and Predicted contours (red)')
    plt.axis('off')

    # Plot the full ground truth mask for the target class
    plt.subplot(2, 2, 2)
    plt.imshow(full_mask, cmap='gray')
    plt.title(f'Ground Truth Mask for Class {target_class}')
    plt.axis('off')

    # Plot the full predicted mask for the target class
    plt.subplot(2, 2, 3)
    plt.imshow(full_pred_mask, cmap='gray')
    plt.title(f'Predicted Mask for Class {target_class}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_attention_with_outlines_and_scales(model, data_loader, device):
    """
    Affiche les cartes d'attention superposées à l'image source avec les organes détourés,
    et inclut une barre d'échelle pour chaque carte d'attention. Les images sont alignées horizontalement.
    La première image affiche aussi la prédiction du modèle en contours bleus.
    Args:
        model: Le modèle PyTorch générant les cartes d'attention.
        data_loader: DataLoader contenant les données (images et masques de labels).
        device: Périphérique ('cuda' ou 'cpu').
    """
    # Mettre le modèle en mode évaluation
    model.eval()

    # Récupérer un batch de données du DataLoader
    data_iter = iter(data_loader)
    random_index = random.randint(0, len(data_loader) - 1)
    for _ in range(random_index):
        next(data_iter)  # Ignorer jusqu'à l'index aléatoire

    input_tensor, labels = next(data_iter)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    labels = labels.squeeze(1).to(device, dtype=torch.long)  # Assurez-vous que les labels sont au bon format

    # Passage des données dans le modèle pour générer les cartes d'attention
    with torch.no_grad():
        output, attention_maps = model(input_tensor, return_attention=True)
        predicted_classes = torch.argmax(output, dim=1)

    # Affichage pour la première image du batch (index 0)
    image_index = 0
    original_image = input_tensor[image_index].cpu().squeeze(0).numpy()
    label_image = labels[image_index].cpu().numpy()
    predicted_image = predicted_classes[image_index].cpu().numpy()

    # Ajouter les contours des labels et des prédictions
    image_with_labels = mark_boundaries(original_image, label_image, color=(1, 1, 0))  # Contours jaunes
    image_with_predictions = mark_boundaries(image_with_labels, predicted_image, color=(0, 0, 1))  # Contours bleus

    # Créer une figure pour aligner les images horizontalement
    num_attention_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_attention_maps + 1, figsize=(20, 5))

    # Afficher l'image originale avec les labels et la prédiction
    axes[0].imshow(image_with_predictions, cmap='gray')
    axes[0].set_title('Image, Labels (Yellow) and Prediction (Blue)')
    axes[0].axis('off')

    # Afficher les cartes d'attention
    for i, attention_map in enumerate(attention_maps):
        # Normaliser la carte d'attention
        att_map = attention_map[image_index].squeeze(0).cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

        # Ajouter l'image originale et la carte d'attention
        im = axes[i + 1].imshow(att_map, cmap='jet')  # Superposition de la carte d'attention

        # Ajouter une barre d'échelle (légende)
        cbar = plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
        cbar.set_label('Visualisation of the Attention', rotation=270, labelpad=15)

        axes[i + 1].set_title(f'Attention Map {i + 1}')
        axes[i + 1].axis('off')

    # Réglage de la mise en page
    plt.subplots_adjust(wspace=0.2)  # Réduction de l'espace entre les images
    plt.tight_layout()
    plt.show()

def display_random_prediction_two_models(model1, model2, val_loader, device):
    """
    Affiche une prédiction aléatoire comparant deux modèles sur un batch de validation.
    Args:
        model1: Premier modèle PyTorch.
        model2: Deuxième modèle PyTorch.
        val_loader: DataLoader contenant les données de validation.
        device: Périphérique ('cuda' ou 'cpu').
    """
    # Mettre les modèles en mode évaluation
    model1.eval()
    model2.eval()

    # Sélectionner un batch aléatoire du valid loader
    data_iter = iter(val_loader)
    random_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_index):
        next(data_iter)  # Ignorer jusqu'à l'index aléatoire

    # Récupérer un batch de données
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Faire des prédictions pour les deux modèles
    with torch.no_grad():
        prediction1 = model1(data)
        predicted_classes1 = torch.argmax(prediction1, dim=1)

        prediction2 = model2(data)
        predicted_classes2 = torch.argmax(prediction2, dim=1)

    # Choisir une image aléatoire dans le batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()
    predicted_image1 = predicted_classes1[random_image_index].cpu().numpy()
    predicted_image2 = predicted_classes2[random_image_index].cpu().numpy()

    # Normalisation de l'image source pour affichage
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Ajouter les contours à l'image source
    image_with_contours = mark_boundaries(data_image, mask_image, color=(0, 1, 0))  # Contours verts (vérité terrain)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image1, color=(0, 0, 1))  # Contours bleus (modèle 1)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image2, color=(1, 0, 0))  # Contours rouges (modèle 2)

    # Afficher l'image avec les contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('True Mask (Vert),  Classic Unet (Bleu), Augmented Unet (Rouge)')
    plt.axis('off')
    plt.show()

def display_comparison(models, val_loader, device, class_names=None):
    """
    Displays a random image with the ground truth mask and predictions from two models.

    Args:
        models: List of trained models [model_1, model_2].
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        class_names: List of class names corresponding to the indices, optional.
    """
    # Set both models to evaluation mode
    for model in models:
        model.eval()

    # Select a random batch from the validation DataLoader
    data_iter = iter(val_loader)
    random_batch_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_batch_index):
        next(data_iter)  # Skip to the selected random batch

    # Get a batch of data and its masks
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Select a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Get predictions from both models
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(data)
            predicted_classes = torch.argmax(prediction, dim=1)
            predictions.append(predicted_classes[random_image_index].cpu().numpy())

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Plot ground truth and predictions
    plt.figure(figsize=(15, 5))

    # Plot the input image
    plt.subplot(1, len(models) + 2, 1)
    plt.imshow(data_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Plot the ground truth mask
    plt.subplot(1, len(models) + 2, 2)
    plt.imshow(mask_image, cmap='viridis')
    if class_names:
        plt.title('Ground Truth')
    else:
        plt.title('Ground Truth Mask')
    plt.axis('off')

    # Plot predictions for each model
    for i, prediction in enumerate(predictions):
        plt.subplot(1, len(models) + 2, i + 3)
        plt.imshow(prediction, cmap='viridis')
        title = f'Prediction (Model {i + 1})'
        if class_names:
            title += f'\n{class_names}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Faire une fonctionn pour plot les deux models sur un meme masque et pour comparer
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="UNet Model Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model 1 (.pth file).")
    parser.add_argument('--model_path2', type=str, required=True, help="Path to the trained model 2 (.pth file).")
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
    print(f"Loading model 1 from {args.model_path}...")
    model =UNet2d()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    print("Evaluation of your first model (Classic UNET)")

    # Define loss function
    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    # Data loading
    print(f"Loading data from {args.data_dir}...")
    # Create dataset and DataLoader
    train_loader,test_loader,val_loader=dataLoaderMaking(namefile=args.data_dir,target_shape = (256, 256),batch_size = args.batch_size)

    # Perform evaluations
    print("Evaluating model...")
    evaluation(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Evaluating with Dice scores...")
    evaluation_with_dice(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Generating confusion matrix...")
    evaluate_confusion_matrix(model, test_loader, device, num_classes=args.num_classes)

    print("Displaying random predictions...")
    display_random_prediction(model, test_loader, device)

    print("Displaying predictions for each class...")
    print("Class 1")
    display_prediction_for_class(model,test_loader,device,1)
    print("Class 2")
    display_prediction_for_class(model,test_loader,device,1)
    print("Class 3")
    display_prediction_for_class(model,test_loader,device,1)
    print("Class 4")
    display_prediction_for_class(model,test_loader,device,1)


    # Load the model 2
    print(f"Loading model 2 from {args.model_path2}...")
    model2=UNetAug2D 
    model2.load_state_dict(torch.load(args.model_path2))
    model2 = model2.to(device)
    print("Evaluation of your first model (Augmented UNET)")

    # Define loss function
    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    # Data loading
    print(f"Loading data from {args.data_dir}...")

    # Perform evaluations
    print("Evaluating model...")
    evaluation(model2, test_loader, criterion, device, num_classes=args.num_classes)

    print("Evaluating with Dice scores...")
    evaluation_with_dice(model2, test_loader, criterion, device, num_classes=args.num_classes)

    print("Generating confusion matrix...")
    evaluate_confusion_matrix(model2, test_loader, device, num_classes=args.num_classes)

    print("Displaying random predictions...")
    display_random_prediction(model2, test_loader, device)

    print("Displaying predictions for each class...")
    print("Class 1")
    display_prediction_for_class(model2,test_loader,device,1)
    print("Class 2")
    display_prediction_for_class(model2,test_loader,device,1)
    print("Class 3")
    display_prediction_for_class(model2,test_loader,device,1)
    print("Class 4")
    display_prediction_for_class(model2,test_loader,device,1)

    print("Displaying Activation Map")
    visualize_attention_with_outlines_and_scales(model2, test_loader, device)

    print("Comparaison between the two models")

    print("Displaying random predictions of the two models...")
    display_random_prediction_two_models(model, model2, test_loader, device)

    print("Displaying Comparaison between the two models")
    display_comparison([model,model2], val_loader, device, class_names=None)

