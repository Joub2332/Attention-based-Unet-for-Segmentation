# Importing libraries 
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import glob
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import random
import itertools
from sklearn.metrics import confusion_matrix
from skimage.segmentation import mark_boundaries


def evaluation(model, val_loader, criterion, device, num_classes=5):
    val_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()  # Mettre le modèle en mode évaluation
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Suppression de la dimension de canal si nécessaire

        with torch.no_grad():
            output = model(data)

        # Calcul de la perte
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Prédictions par pixel
        _, preds = torch.max(output, dim=1)

        # Calcul des pixels corrects pour chaque classe
        for i in range(num_classes):
            class_mask = (label == i)
            class_correct[i] += torch.sum((preds == i) & class_mask).item()
            class_total[i] += torch.sum(class_mask).item()

    # Calcul de la perte moyenne
    val_loss /= len(val_loader.dataset)

    # Affichage des précisions par classe
    print('Validation Loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy_class = 100.0 * class_correct[i] / class_total[i]
            print(f'Accuracy for class {i}: {accuracy_class:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Accuracy for class {i}: N/A (no pixels)')

    # Calcul de la précision globale
    total_correct_pixels = sum(class_correct)
    total_pixels = sum(class_total)
    overall_accuracy = 100.0 * total_correct_pixels / total_pixels

    print('Overall pixel-wise accuracy: {:.2f}% ({}/{})'.format(overall_accuracy, total_correct_pixels, total_pixels))

    return val_loss, overall_accuracy, class_correct, class_total


def evaluation_with_dice(model, val_loader, criterion, device, num_classes=5):
    val_loss = 0.0
    dice_scores = [0.0] * num_classes  # Dice score par classe
    class_total = [0] * num_classes

    model.eval()  # Mettre le modèle en mode évaluation
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Suppression de la dimension de canal si nécessaire

        with torch.no_grad():
            output = model(data)

        # Calcul de la perte
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Prédictions par pixel
        _, preds = torch.max(output, dim=1)

        # Calcul du Dice Score pour chaque classe
        for i in range(num_classes):
            class_mask = (label == i)
            pred_mask = (preds == i)

            # Intersection et union pour le Dice Score
            intersection = torch.sum(pred_mask & class_mask).item()
            union = torch.sum(pred_mask).item() + torch.sum(class_mask).item()

            if union > 0:  # Éviter une division par zéro
                dice_scores[i] += 2.0 * intersection / union
                class_total[i] += 1

    # Calcul de la perte moyenne
    val_loss /= len(val_loader.dataset)

    # Affichage des scores Dice par classe
    print('Validation Loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            average_dice = dice_scores[i] / class_total[i]
            print(f'Dice Score for class {i}: {average_dice:.4f}')
        else:
            print(f'Dice Score for class {i}: N/A (no pixels)')

    # Calcul du Dice Score moyen global
    global_dice = sum(dice_scores) / sum(class_total)
    print('Overall Dice Score: {:.4f}'.format(global_dice))

    return val_loss, global_dice, dice_scores

def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice de Confusion', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matrice de confusion normalisée")
    else:
        print('Matrice de confusion sans normalisation')

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
    plt.ylabel('Étiquette Vraie')
    plt.xlabel('Étiquette Prédite')

def evaluate_confusion_matrix(model, val_loader, device, num_classes=5):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for data, mask in val_loader:
            data = data.to(device)
            mask = mask.squeeze(1).to(device).long()  # Conversion explicite en entiers

            # Prédictions du modèle
            output = model(data)
            predicted_classes = torch.argmax(output, dim=1).long()  # Conversion explicite en entiers

            # Collecter les valeurs réelles et prédites
            y_true.extend(mask.cpu().numpy().flatten().astype(int))  # Conversion explicite en entiers
            y_pred.extend(predicted_classes.cpu().numpy().flatten().astype(int))  # Conversion explicite en entiers

    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Afficher la matrice de confusion
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

    # Convertir en format CPU pour l'affichage
    data = data.cpu().squeeze(0).squeeze(0).numpy()  # Première image du batch
    mask = mask.cpu().squeeze(0).numpy()            # Vérité terrain
    predicted_classes = predicted_classes.cpu().squeeze(0).numpy()  # Prédictions

    # Normalisation de l'image source pour affichage
    data = (data - data.min()) / (data.max() - data.min())

    # Ajouter les contours à l'image source
    image_with_contours = mark_boundaries(data, mask, color=(0, 1, 0))  # Contours verts (vérité terrain)
    image_with_contours = mark_boundaries(image_with_contours, predicted_classes, color=(1, 0, 0))  # Contours rouges (prédictions)

    # Afficher l'image avec les contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('Contours Vrais (Vert) et Prédits (Rouge)')
    plt.axis('off')
    plt.show()

#Faire une fonctionn pour plot les deux models sur un meme masque et pour comparer