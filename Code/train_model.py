import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import random

# --- Hyperparameters ---
SYNTHETIC_DIR = "final_dataset_safe" # Path to query images (noisy)
CATALOG_DIR = "catalog"             # Path to anchor images (clean)
BATCH_SIZE = 8                      # ResNet50 requires significant VRAM
EPOCHS = 10
LEARNING_RATE = 0.0001
EMBEDDING_DIM = 128                 # Size of the vector representing the image

# --- 1. Custom Triplet Dataset ---
class TripletVinylDataset(Dataset):
    """
    Returns (Anchor, Positive, Negative) triplets for contrastive learning.
    Anchor: Noisy CCTV image.
    Positive: Clean catalog version of the same album.
    Negative: Clean catalog image of a DIFFERENT album.
    """
    def __init__(self, synthetic_dir, catalog_dir, transform=None):
        self.synthetic_dir = synthetic_dir
        self.catalog_dir = catalog_dir
        self.transform = transform
        self.synthetic_images = [f for f in os.listdir(synthetic_dir) if f.endswith('.jpg')]
        self.catalog_images = [f for f in os.listdir(catalog_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.synthetic_images)

    def __getitem__(self, idx):
        # A. Anchor (CCTV Query)
        anchor_filename = self.synthetic_images[idx]
        anchor_path = os.path.join(self.synthetic_dir, anchor_filename)
        anchor_img = Image.open(anchor_path).convert("RGB")

        # B. Positive (Matching Clean Image)
        clean_name = anchor_filename.split("_v")[0] 
        positive_filename = next((c for c in self.catalog_images if os.path.splitext(c)[0] == clean_name), random.choice(self.catalog_images))
        positive_path = os.path.join(self.catalog_dir, positive_filename)
        positive_img = Image.open(positive_path).convert("RGB")

        # C. Negative (Different Clean Image)
        negative_filename = positive_filename
        while negative_filename == positive_filename:
            negative_filename = random.choice(self.catalog_images)
        negative_path = os.path.join(self.catalog_dir, negative_filename)
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img, positive_img, negative_img = self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# --- 2. Siamese Embedding Network (ResNet50) ---
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Use ResNet50 for high-capacity feature extraction
        self.backbone = models.resnet50(weights='DEFAULT')
        num_ftrs = self.backbone.fc.in_features # 2048 for ResNet50
        
        # Replace the classifier with an embedding head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim) # LayerNorm helps converge faster with Triplet Loss
        )

    def forward(self, x):
        return self.backbone(x)

# --- 3. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on: {device}")

    # Standard normalization for ImageNet-pretrained backbones
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TripletVinylDataset(SYNTHETIC_DIR, CATALOG_DIR, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)

    # TripletMarginLoss: minimizes distance (A, P) and maximizes distance (A, N)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            
            # Map all images into the same embedding space
            a_emb, p_emb, n_emb = model(anchor), model(positive), model(negative)

            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    # Save the final weights for evaluation
    torch.save(model.state_dict(), "vinyl_model_final.pth")
    print("🎉 Training Complete. Model saved.")

if __name__ == "__main__":
    train()