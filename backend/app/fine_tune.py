"""
Fine-tune the model on incorrectly predicted samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from app.models import BacteriaDetectorCNN
import pickle


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning on hard examples"""
    
    def __init__(self, image_patch_pairs):
        """
        Args:
            image_patch_pairs: List of (patch, label) tuples
                patch: 48x48x3 numpy array
                label: 0 (bacteria) or 1 (background)
        """
        self.data = image_patch_pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patch, label = self.data[idx]
        
        # Normalize
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor


def fine_tune_model(
    model_path: str,
    train_patches: list,
    val_patches: list,
    output_path: str,
    num_epochs: int = 10,
    learning_rate: float = 0.0001
):
    """
    Fine-tune model on hard examples
    
    Args:
        model_path: Path to current model
        train_patches: List of (patch, label) for training
        val_patches: List of (patch, label) for validation
        output_path: Where to save fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate (should be small for fine-tuning)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load existing model
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    model = BacteriaDetectorCNN()
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Create datasets
    train_dataset = FineTuneDataset(train_patches)
    val_dataset = FineTuneDataset(val_patches)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    print(f"\nFine-tuning for {num_epochs} epochs...")
    print(f"Training samples: {len(train_patches)}")
    print(f"Validation samples: {len(val_patches)}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(output_path, 'wb') as f:
                pickle.dump(model.state_dict(), f)
            print(f"  ✅ Saved improved model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Saved to: {output_path}")


# Example usage:
if __name__ == "__main__":
    # You need to manually create these patches from your images
    # where the model made mistakes
    
    # Example: patches where model was wrong
    train_patches = [
        # (48x48x3 numpy array, correct_label)
        # Add false positives with label=1 (background)
        # Add false negatives with label=0 (bacteria)
    ]
    
    val_patches = [
        # Validation patches
    ]
    
    fine_tune_model(
        model_path="app/model/final_bacteria_detector_weights.pkl",
        train_patches=train_patches,
        val_patches=val_patches,
        output_path="app/model/fine_tuned_model.pkl",
        num_epochs=10,
        learning_rate=0.0001
    )