import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# --- Configuration & Hyperparameters ---
CATALOG_DIR = "catalog"                # Directory containing clean reference images
TEST_DIR = "test_dataset_unseen"       # Directory containing degraded CCTV query images
MODEL_PATH = "vinyl_model_final.pth"   # Path to the trained ResNet50 weights
EMBEDDING_DIM = 128                    # Dimension of the final feature vector
TOP_K = 5                              # We evaluate if the correct album is in the top 5 matches

# --- Model Definition (Architecture must match the training script) ---
class EmbeddingNet(nn.Module):
    """
    Siamese Branch using ResNet50 as a backbone.
    Final FC layers are modified to produce a 128-dim embedding.
    """
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Initialize ResNet50 without pretrained weights (we load our own later)
        self.backbone = models.resnet50(weights=None) 
        
        # ResNet50 output features before the FC layer is 2048
        num_ftrs = self.backbone.fc.in_features 
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),         # Intermediate layer for better feature representation
            nn.ReLU(),
            nn.Linear(512, embedding_dim),    # Final projection to embedding space
            nn.LayerNorm(embedding_dim)       # Normalization helps with Triplet Loss stability
        )

    def forward(self, x):
        return self.backbone(x)

def evaluate():
    # Setup device (GPU is highly recommended for inference speed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on: {device}")

    # Standard ImageNet normalization used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Load the trained model
    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("✅ Model loaded successfully.")
        except RuntimeError as e:
            print(f"❌ Error: Architecture mismatch. Ensure ResNet50 is used: {e}")
            return
    else:
        print("❌ Error: Model file not found! Please run train_model.py first.")
        return
    
    model.eval() # Set to evaluation mode (disables dropout/batchnorm updates)

    # 2. Build the Reference Catalog Index
    # We pre-calculate embeddings for all clean images in the catalog
    print("📚 Building Catalog Index...")
    catalog_vectors = []
    catalog_names = []
    
    catalog_files = [f for f in os.listdir(CATALOG_DIR) if f.lower().endswith('.jpg')]
    
    with torch.no_grad(): # No gradient calculation needed for inference
        for f in tqdm(catalog_files, desc="Indexing Catalog"):
            img_path = os.path.join(CATALOG_DIR, f)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(img_tensor).cpu().numpy()
            
            catalog_vectors.append(embedding)
            catalog_names.append(f)

    # Convert list of vectors into a matrix for efficient distance calculation
    catalog_matrix = np.vstack(catalog_vectors)

    # 3. Performance Testing (Precision @ K)
    # Testing on 'Unseen' images to verify generalization capability
    print(f"\n🚀 Starting Evaluation (Top-{TOP_K})...")
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith('.jpg')]
    test_files = test_files[:100] # Standardized test on 100 images as per report

    correct_predictions = 0
    total_checked = 0

    with torch.no_grad():
        for test_f in tqdm(test_files, desc="Testing CCTV Images"):
            img_path = os.path.join(TEST_DIR, test_f)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            query_vec = model(img_tensor).cpu().numpy()

            # Calculate L2 distance (Euclidean) between query and all catalog items
            dists = np.linalg.norm(catalog_matrix - query_vec, axis=1)
            
            # Retrieve the indices of the Top-K closest matches
            closest_indices = np.argsort(dists)[:TOP_K]
            top_k_names = [catalog_names[i] for i in closest_indices]

            # Extract ground truth name (removing version suffix like _v01.jpg)
            true_name_clean = test_f.split("_v")[0]
            
            # Check if the correct album is within the retrieved results
            found = False
            for prediction in top_k_names:
                if true_name_clean in prediction:
                    found = True
                    break
            
            if found:
                correct_predictions += 1
            
            total_checked += 1

    # Final Accuracy calculation (Precision at K)
    accuracy = (correct_predictions / total_checked) * 100
    print("-" * 30)
    print(f"📊 Final Results (ResNet50):")
    print(f"Tested on {total_checked} images.")
    print(f"Precision{TOP_K}: {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()