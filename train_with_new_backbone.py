import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from dataloader.MultiBackboneDataset import create_dataloaders
from core.models.er_former import ErrorRecognitionTransformer  # Use your existing model
from core.evaluate import evaluate_model  # Use your existing evaluation

def train_model(backbone='egovlp', epochs=50):
    """Train model with specified backbone"""
    
    print(f"\nTraining with {backbone.upper()} features")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features_dir='./extracted_features',
        backbone=backbone,
        batch_size=16
    )
    
    # Get feature dimension
    feature_dims = {
        'omnivore': 1024,
        'slowfast': 2304,
        'egovlp': 512,
    }
    feature_dim = feature_dims[backbone]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ErrorRecognitionTransformer(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            video_features = batch['video_features'].to(device)
            labels = batch['labels'].float().to(device)
            masks = batch['segment_masks'].to(device)
            
            optimizer.zero_grad()
            logits = model(video_features, masks)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), f'best_model_{backbone}.pth')
    
    # Test
    model.load_state_dict(torch.load(f'best_model_{backbone}.pth'))
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    return test_metrics


def evaluate_model(model, dataloader, device):
    """Evaluate model - use your existing implementation"""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            video_features = batch['video_features'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['segment_masks'].to(device)
            
            logits = model(video_features, masks)
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }


if __name__ == "__main__":
    # Train with EgoVLP
    train_model(backbone='egovlp', epochs=50)