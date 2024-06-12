import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score , confusion_matrix
import tqdm
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, train_dir: str, augment : bool =False, noise_factor : float =0.1):
        """
        Args:
            train_dir (string): Directory with 'X.csv' and 'targets.csv' files.
            augment (bool): Whether to apply augmentation.
            noise_factor (float): Factor to control the amount of noise added for augmentation.
        """
        # Construct file paths
        data_path = os.path.join(train_dir, "X.csv")
        target_path = os.path.join(train_dir, "targets.csv")
        
        # Load the data and targets from csv files
        self.data = pd.read_csv(data_path)
        self.targets = pd.read_csv(target_path)
        
        # Ensure the data and targets have matching lengths
        assert len(self.data) == len(self.targets), "Data and targets must have the same length"
        
        # Augmentation parameters
        self.augment = augment
        self.noise_factor = noise_factor
        
    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the features and target for the given index
        data_sample = self.data.iloc[idx].values
        target_sample = self.targets.iloc[idx].values
        
        # Apply augmentation if specified
        if self.augment:
            data_sample = self._augment_data(data_sample)
        
        # Convert to torch tensors
        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        target_sample = torch.tensor(target_sample, dtype=torch.float32)
        
        return data_sample, target_sample
    
    def _augment_data(self, data_sample):
        # Add random noise to the data sample
        np.random.seed(42)
        noise = np.random.normal(0, self.noise_factor, data_sample.shape)
        augmented_data = data_sample + noise
        return augmented_data
    
def train_step(model: torch.nn.Module,
               train_dl: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float]:

    model.to(device)
    model.train()

    train_accuracy = 0.0
    train_loss = 0.0

    for idx, (X_train, train_targets) in enumerate(train_dl):

        X_train, train_targets = X_train.to(device), train_targets.to(device)

        # Ensure targets are of type LongTensor
        train_targets = train_targets.long()

        train_logits = model(X_train)
        train_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)

        # Compute loss
        loss = loss_fn(train_logits, train_targets.squeeze(1))

        # Calculate accuracy
        train_acc = accuracy_score(train_targets.cpu(), train_preds.cpu())

        train_accuracy += train_acc
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy /= len(train_dl)
    train_loss /= len(train_dl)

    return train_accuracy, train_loss

def eval_step(model: torch.nn.Module,
              val_dl: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> tuple[float, float]:


    val_accuracy = 0.0
    val_loss = 0.0

    model.to(device)
    model.eval()

    with torch.inference_mode():

        for X_val, val_targets in val_dl:
            X_val, val_targets = X_val.to(device), val_targets.to(device)
            
            val_targets = val_targets.long()
            
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, val_targets.squeeze(dim=1))

            val_preds = torch.softmax(val_logits,dim=1).argmax(dim=1).unsqueeze(1)
            val_acc = accuracy_score(val_targets.flatten() , val_preds.flatten())

            val_accuracy += val_acc
            val_loss += val_loss.item()

        val_accuracy /= len(val_dl)
        val_loss /= len(val_dl)

    return val_accuracy, val_loss

def train_model(model : torch.nn.Module,
               train_dl : torch.utils.data.DataLoader,
               val_dl : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               epochs: int,
               device : torch.device) -> dict[str, list[float]]:
    

    results = {"train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": [],
    }
    
    for epoch in tqdm(range(epochs)):
        
        train_accuracy , train_loss =train_step(train_dl=train_dl, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            device = device
        )

        val_accuracy , val_loss =eval_step(
            model=model,
            val_dl= val_dl,
            loss_fn=loss_fn,
            device = device
        )

        results['train_accuracy'].append(train_accuracy)
        results['train_loss'].append(train_loss)
        results['val_accuracy'].append(val_accuracy)
        results['val_loss'].append(val_loss)

        print('-------------- EPOCH {} ----------------------------'.format(epoch))
        print(f"Train Accuracy: {train_accuracy:.5f} | Train Loss : {train_loss:.5f}")
        print(f"Val Accuracy: {val_accuracy:.5f} | Val Loss: {val_loss:.5f}\n")
        print()

    return results

from matplotlib.gridspec import GridSpec
def plot_loss_curve_grid(results: dict[str, list[float]]) -> None:
    """
    Plots training curves of a results dictionary.
    Args:
        results (dict): Dictionary containing lists of values.
            Example:
            {'loss_generator': [float],
             'loss_discriminator': [float],
             'real_scores': [float],
             'fake_scores': [float]}
    Plots:
        - Generator Loss over epochs.
        - Discriminator Loss over epochs.
        - Real Score over epochs.
        - Fake Score over epochs.
    """

    train_accuracy = results['train_accuracy']
    train_loss = results['train_loss']
    val_accuracy = results['val_accuracy']
    val_loss = results['val_loss']

    epochs = range(len(train_accuracy))

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_accuracy, marker='o', color='b', label='Train Accuracy')
    ax1.set_title('Train Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_loss, marker='s', color='r', label='Train Loss')
    ax2.set_title('Train Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_accuracy, marker='o', color='g', label='Val Accuracy')
    ax3.set_title('Val Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, val_loss, marker='s', color='purple', label='Val Loss')
    ax4.set_title('Val Loss')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

def plot_loss_curves(results: dict[str, list[float]]) -> None:
    """
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {'loss_generator' : [],
            'loss_discriminator' : [],
            'real_scores' : [],
            'fake_scores' : [],}
    """
    train_accuracy = results['train_accuracy']
    train_loss = results['train_loss']
    val_accuracy = results['val_accuracy']
    val_loss = results['val_loss']

    epochs = range(len(results['train_accuracy']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, val_accuracy, label='Val Accuracy')
    plt.title('Accuracy : Train vs Val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss : Train vs Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


def get_preds(model:torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              device : torch.device) -> list[torch.Tensor]:
    """
    Generate predictions using the provided model and data loader.

    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform predictions on (e.g., 'cuda' or 'cpu').

    Returns:
        list[torch.Tensor]: A list containing the predictions for each sample in the dataset.
    """

    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            X, y = X.to(device), y.to(device)
            y = y.long()

            y_logits = model(X)
            y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)

            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

    return y_pred_tensor

def plot_confusion_matrix(title: str ,targets:torch.Tensor , preds : torch.Tensor):
    cf = confusion_matrix(targets, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cf , annot=True , cmap = 'Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'Confusion Matrix : {title}') 
    plt.show()