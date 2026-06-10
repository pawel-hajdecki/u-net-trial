import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import UNet

# 1. Setup device and load your trained architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(n_channels=3, n_classes=3).to(device)  # Make sure this matches your model definition

# 2. Load the saved weights (checkpoint)
print("=> Loading checkpoint...")
checkpoint = torch.load("checkpoint.pth.tar", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()  # Put model in evaluation mode!

# 3. Load and transform your single test image
image_path = "data/train_images/Ragdoll_171.jpg"  # Swap this out with any image path! 
original_image = Image.open(image_path).convert("RGB")

# Match the exact preprocessing used during your training stage
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Change to match your training normalization
])

# Add a batch dimension: [3, 256, 256] becomes [1, 3, 256, 256]
input_tensor = transform(original_image).unsqueeze(0).to(device)

# # 4. Predict the mask!
# with torch.no_grad():
#     logits = model(input_tensor)
#     probs = torch.sigmoid(logits)
#     # Threshold at 0.5: anything above becomes 255 (White), below becomes 0 (Black)
#     pred_mask = (probs > 0.5).float() * 255.0

with torch.no_grad():
    logits = model(input_tensor)
    # 2. Use argmax across the channel dimension (dim=1) for multiclass data
    pred_mask = torch.argmax(logits, dim=1).squeeze(0) # Shape becomes [256, 256]

# 5. Convert back to an image and save it to disk
pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8) # Remove batch/channel dims
visble_mask = pred_mask * 80
output_image = Image.fromarray(visble_mask)
output_image.save("predicted_mask_output.png")

print("Finished! Check your project folder for 'predicted_mask_output.png' to see the prediction.")