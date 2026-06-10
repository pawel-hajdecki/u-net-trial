import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import SegmentationDataset
from utils import save_checkpoint, evaluate_model
from PIL import Image
import numpy as np

# Configurable Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DIR = "data/train_images/"
MASK_DIR = "data/train_masks/"

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device).long() # convert to int for cross entropy 

        if len(targets.shape) == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # Forward pass
        outputs = model(data) # Shape: [Batch, 3, 256, 256]
        loss = loss_fn(outputs, targets) # Shape: [Batch, 256, 256]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Loss: {running_loss / len(loader):.4f}")

def main():
    # Initialize Core Objects
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Data Infrastructure
    train_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR, 
        mask_dir=MASK_DIR, 
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    # Execution Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        
        # Run validation metrics
        evaluate_model(train_loader, model, device=DEVICE)
        
        # Save structural checkpoint state
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)


def small_test():
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Data Infrastructure
    train_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR, 
        mask_dir=MASK_DIR, 
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    example_data, example_targets = next(iter(train_loader))
    example_data = example_data.to(DEVICE)
    example_targets = example_targets.to(DEVICE).long()
    if len(example_targets.shape) == 4 and example_targets.shape[1] == 1:
        example_targets = example_targets.squeeze(1)

    print("--- Starting Sanity Check: Overfitting One Batch ---")
    model.train()
    for step in range(50):
        outputs = model(example_data)
        loss = loss_fn(outputs, example_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}/50 | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        # Pass just the very first image from our batch
        single_input = example_data[0].unsqueeze(0) # Keep batch dim [1, 3, 256, 256]
        logits = model(single_input)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0) # Shape: [256, 256]

    # Convert to a visible image
    pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
    visible_mask = pred_mask_np * 100 # Scale classes (0, 1, 2) to (0, 100, 200)

    output_image = Image.fromarray(visible_mask)
    output_image.save("sanity_prediction.png")

if __name__ == "__main__":
    # small_test()
    main()