# --- Grid & Detection Parameters ---
S_W = 20                   # Grid cells horizontally (IMAGE_WIDTH / 32)
S_H = 13                   # Grid cells vertically (IMAGE_HEIGHT / 32)
C = 11                     # Number of classes 
B = 1

# --- Input Image Dimensions ---
IMAGE_HEIGHT=416
IMAGE_WIDTH=640

# --- YOLO Loss Scaling Weights ---
LAMBDA_COORD = 5.0         
LAMBDA_NOOBJ = 0.5         

# --- Inference Thresholds ---
CONF_THRESHOLD = 0.25      # Score threshold to keep box candidates during decoding
NMS_IOU_THRESH = 0.4       # IoU threshold for Non-Maximum Suppression (NMS)

# --- Training Hyperparameters ---
LR = 1e-4   
WEIGHT_DECAY=1e-4
EPOCHS = 75                
BATCH_SIZE = 64            

# --- Hardware Loader Settings ---
NUM_WORKERS = 2            # Number of CPU background processes loading data
PIN_MEMORY = True          # Pin loaded tensors to CPU RAM for faster GPU transfer

# --- Paths ---
DATASET_ROOT = "dataset/"  

# Save checkpoints directly to permanent Google Drive
CHECKPOINT_DIR = "/content/drive/MyDrive/yolov1/checkpoints"
LOG_DIR = "/content/drive/MyDrive/yolov1/runs"