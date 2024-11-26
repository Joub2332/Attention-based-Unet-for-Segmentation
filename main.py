# Script principal
import argparse
import torch
import torch.nn as nn
from model import UNet2d, UNetAug2D
from data import dataLoaderMaking
from evaluation import evaluation, evaluation_with_dice, evaluate_confusion_matrix, display_random_prediction
from train import training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate U-Net Models")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--train', action='store_true', help="Specify to train models")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--load_classic', type=str, default=None, help="Path to classic U-Net model")
    parser.add_argument('--load_aug', type=str, default=None, help="Path to augmented U-Net model")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, test_loader, val_loader = dataLoaderMaking(args.dataset_path, target_shape=(256, 256), batch_size=args.batch_size)
    class_weights = torch.tensor([1.04, 30.3, 263.1, 158.7, 270.3]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.train:
        # Initialiser les modèles
        model_classic = UNet2d().to(device)
        model_aug = UNetAug2D().to(device)

        # Définir la perte et les optimisateurs
        
        optimizer_classic = torch.optim.Adam(model_classic.parameters(), lr=0.00005)
        optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=0.00005)

        # Entraîner les modèles
        print("Training Classic U-Net...")
        training(model_classic, criterion, optimizer_classic, train_loader, val_loader, device, args.epochs, "model_classic.pt")

        print("Training Augmented U-Net...")
        training(model_aug, criterion, optimizer_aug, train_loader, val_loader, device, args.epochs, "model_aug.pt")

        print("Evaluating Classic U-Net...")
        evaluation(model_classic, test_loader, nn.CrossEntropyLoss(), device)
        evaluation_with_dice(model_classic, test_loader, criterion, device, num_classes=5)
        evaluate_confusion_matrix(model_classic, test_loader, device, num_classes=5):
        display_random_prediction(model_classic, test_loader, device)

        #bout de code suivant à mettre dans une fonction et a généraliser pour pas tout répéter
        print("Evaluating Augmented U-Net...")
        evaluation(model_aug, test_loader, nn.CrossEntropyLoss(), device)
        evaluation_with_dice(model_aug, test_loader, criterion, device, num_classes=5)
        evaluate_confusion_matrix(model_aug, test_loader, device, num_classes=5)
        display_random_prediction(model_aug, test_loader, device)

        print("comparison between the two models...")
        #ajouter la fonction dans évaluation
    else:
        # Charger les modèles si spécifié
        if args.load_classic:
            model_classic = UNet2d().to(device)
            model_classic.load_state_dict(torch.load(args.load_classic, map_location=device))
            print("Evaluating Classic U-Net...")
            evaluation(model_classic, test_loader, nn.CrossEntropyLoss(), device)
            evaluation_with_dice(model_classic, test_loader, criterion, device, num_classes=5)
            evaluate_confusion_matrix(model_classic, test_loader, device, num_classes=5):
            display_random_prediction(model_classic, test_loader, device)

        if args.load_aug:
            model_aug = UNetAug2D().to(device)
            model_aug.load_state_dict(torch.load(args.load_aug, map_location=device))
            print("Evaluating Augmented U-Net...")
            evaluation(model_aug, test_loader, nn.CrossEntropyLoss(), device)
            evaluation_with_dice(model_aug, test_loader, criterion, device, num_classes=5)
            evaluate_confusion_matrix(model_aug, test_loader, device, num_classes=5):
            display_random_prediction(model_aug, test_loader, device)
       
        if args.load_classic and args.load_aug : 
            #ajouter une fonction dans le fichier evalutation pour comparer les deux sur une meme image.