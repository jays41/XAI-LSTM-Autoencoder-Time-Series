import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from model import LSTMAutoencoder


class TimeSeriesDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx]


def train_epoch(model, dataloader, optimiser, criterion, device, max_norm=5.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        optimiser.zero_grad()
        reconstruction, _ = model(batch)
        loss = criterion(reconstruction, batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimiser.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=500,
    lr=0.001,
    patience_early_stop=20,
    patience_lr_scheduler=50,
    lr_factor=0.5,
    checkpoint_path="best_model.pth",
    history_path="training_history.json"
):
    
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=lr_factor, patience=patience_lr_scheduler, verbose=True
    )
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
        
        # early stopping
        if epochs_no_improve >= patience_early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def plot_loss_curves(history, save_path="loss_curves.png"):
    """Plot and save training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['lr'], color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, latent = model(batch)
            latents.append(latent.cpu().numpy())
    
    return np.vstack(latents)


def extract_reconstruction_errors(model, dataloader, device):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            mse = ((reconstruction - batch) ** 2).mean(dim=[1, 2])
            errors.append(mse.cpu().numpy())
    
    return np.concatenate(errors)


def main():
    device = torch.device('cpu')
    batch_size = 64
    num_workers = 0  # set to 0 on Windows to avoid multiprocessing issues
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    train_windows = np.load("train_windows.npy")
    val_windows = np.load("val_windows.npy")
    test_windows = np.load("test_windows.npy")
    
    train_dataset = TimeSeriesDataset(train_windows)
    val_dataset = TimeSeriesDataset(val_windows)
    test_dataset = TimeSeriesDataset(test_windows)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    input_dim = train_windows.shape[2]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        dropout=0.2
    )
    model = model.to(device)
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=500,
        lr=0.001,
        patience_early_stop=20,
        patience_lr_scheduler=50,
        lr_factor=0.5,
        checkpoint_path=output_dir / "best_model.pth",
        history_path=output_dir / "training_history.json"
    )
    
    plot_loss_curves(history, save_path=output_dir / "loss_curves.png")
    
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_latents = extract_latents(model, train_loader, device)
    val_latents = extract_latents(model, val_loader, device)
    test_latents = extract_latents(model, test_loader, device)
    
    np.save(output_dir / "train_latents.npy", train_latents)
    np.save(output_dir / "val_latents.npy", val_latents)
    np.save(output_dir / "test_latents.npy", test_latents)
    
    print(f"Train latents: {train_latents.shape}")
    print(f"Val latents:   {val_latents.shape}")
    print(f"Test latents:  {test_latents.shape}")
    
    train_errors = extract_reconstruction_errors(model, train_loader, device)
    val_errors = extract_reconstruction_errors(model, val_loader, device)
    test_errors = extract_reconstruction_errors(model, test_loader, device)
    
    np.save(output_dir / "train_errors.npy", train_errors)
    np.save(output_dir / "val_errors.npy", val_errors)
    np.save(output_dir / "test_errors.npy", test_errors)
    
    print(f"Train errors: {train_errors.shape} | Mean: {train_errors.mean():.6f} | Std: {train_errors.std():.6f}")
    print(f"Val errors:   {val_errors.shape} | Mean: {val_errors.mean():.6f} | Std: {val_errors.std():.6f}")
    print(f"Test errors:  {test_errors.shape} | Mean: {test_errors.mean():.6f} | Std: {test_errors.std():.6f}")


if __name__ == "__main__":
    main()