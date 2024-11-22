import argparse
import nbimporter
from dataPrepare.ipynb import 
#Fonction pour lire les arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the training dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    return parser.parse_args()



# Fonction d'entraînement modifiée
def training(model, criterion, optimizer, train_loader, val_loader,device,n_epochs):
    numberSamples = len(train_loader.dataset)
    train_losses, valid_losses = [], []
    valid_loss_min = np.inf
    i = 1
    
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0, 0

        # Entraînement
        model.train()
        for data, label in train_loader:
            data = data.to(device)  # Ajouter la dimension des canaux
            label = label.squeeze(1).to(device).long()

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            print(epoch,i,":",train_loss)
            i += 1
            

        # Validation
        model.eval()
        for data, label in val_loader:
            data = data.to(device)  # Ajouter la dimension des canaux
            label = label.squeeze(1).to(device).long()

            with torch.no_grad():
                output = model(data)
            loss = criterion(output, label)
            valid_loss += loss.item() * data.size(0)

        # Calcul des pertes moyennes
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        # Sauvegarder le modèle si la perte de validation a diminué
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), 'modelUnetClassique.pt')
            valid_loss_min = valid_loss

    return train_losses, valid_losses

if __name__ == "__main__":
    args = parse_args()

    # Charger les données
    data_loader = load_data(args.dataset_path)

    # Définir le modèle
    model = define_model()

    # Entraîner le modèle   

    train_losses, valid_losses = training(model, criterion, optimizer, train_loader, val_loader,device)

# Exécuter l'entraînement
