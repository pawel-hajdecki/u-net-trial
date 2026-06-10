import torch

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

def evaluate_model(loader, model, device="cuda"):
    """Evaluates pixel accuracy and Multi-class Dice Score for 3 classes"""
    num_correct = 0
    num_pixels = 0
    total_dice = 0.0
    
    num_classes = 3  # Background, Pet, Border
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()  # Targets must be long integers
            
            # Squeeze channel dimension if it's [B, 1, H, W] -> [B, H, W]
            if len(y.shape) == 4 and y.shape[1] == 1:
                y = y.squeeze(1)
            
            # 1. Multi-class prediction using argmax
            logits = model(x)  # Shape: [B, 3, H, W]
            preds = torch.argmax(logits, dim=1)  # Shape: [B, H, W]
            
            # 2. Calculate Global Pixel Accuracy
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            
            # 3. Calculate Multi-class Dice Score (Macro Averaged)
            batch_dice = 0.0
            for cls in range(num_classes):
                # Isolate current class as a binary mask of 0s and 1s
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()
                
                intersection = (pred_cls * true_cls).sum().item()
                total_elements = (pred_cls + true_cls).sum().item()
                
                # If neither the model predicted nor the target contains this class, 
                # its overlap is technically 100% perfect (1.0)
                if total_elements == 0:
                    batch_dice += 1.0
                else:
                    batch_dice += (2.0 * intersection) / (total_elements + 1e-8)
            
            # Average across all 3 classes for this specific batch
            total_dice += (batch_dice / num_classes)

    final_accuracy = (num_correct / num_pixels) * 100
    final_dice = total_dice / len(loader)

    print(f"Pixel Accuracy: {final_accuracy:.2f}%")
    print(f"Dice Score (Macro): {final_dice:.4f}")
    
    model.train()
    


def visualize_mask(mask_path, output_path="visible_mask_check.png"):
    from PIL import Image
    import numpy as np
    # Load the mask that looks completely black
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # Multiply the low values (1, 2, 3) by 80 to blast them into high-contrast visibility
    visible_mask = Image.fromarray(mask_array * 80)
    visible_mask.save(output_path)