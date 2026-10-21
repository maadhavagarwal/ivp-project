import os

# Project Root Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset Paths
DATASET_DIR = BASE_DIR
CSV_PATH = os.path.join(BASE_DIR, "train.csv")

# Model Directory
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dr_model.pth")
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "dr_video_model.pth")
CLASSICAL_MODEL_PATH = os.path.join(MODEL_DIR, "classical_rf_model.pkl")
RF_MODEL_PATH = CLASSICAL_MODEL_PATH

# Output Directory for Processed Images / Reports
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class Labels Mapping
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

CLASS_FOLDERS = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

CLASS_COLORS = {
    0: "#2ec4b6",  # Teal / Green
    1: "#ffbf00",  # Yellow
    2: "#ff9f1c",  # Orange
    3: "#e63946",  # Red-Orange
    4: "#9d0208"   # Dark Red / Emergency
}

# Image Preprocessing Configuration
IMAGE_SIZE = (224, 224)
BEN_GRAHAM_SIGMA = 10
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Video Processing Configuration
VIDEO_FOCUS_THRESHOLD = 50.0  # Minimum Laplacian variance for frame focus quality
VIDEO_FRAME_STEP = 5          # Process every Nth frame for efficiency
STABILITY_WINDOW = 5          # Number of valid frames for prediction voting

# Model Training Configuration
BATCH_SIZE = 32
NUM_CLASSES = 5
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"  # PyTorch device
