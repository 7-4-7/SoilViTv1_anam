# Soil Classification Challenge

A deep learning solution for soil type classification using PyTorch with a custom TinyVGG architecture. This project provides a complete pipeline for data preprocessing, model training, evaluation, and inference.

##  Project Structure

```
Challenge #1/
├── data/
│   └── download.sh                         # Dataset download script
├── docs/
│   └── cards/
│       ├── ml-metrics.json                 # Model performance metrics
│       ├── tinyvgg_architecture.png        # Model architecture diagram
│       └── project-card.ipynb              # Model project card
├── notebooks/
│   ├── training.ipynb                      # Model training notebook
│   └── inferencing.ipynb                   # Model testing notebook
├── src/
│   ├── __init__.py
│   └── essentials.py                       # Essential functions and utilities
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
└── LICENSE                                # MIT License
```

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Kaggle account for dataset access

### 1. Environment Setup

**Create and activate a virtual environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Installation

**Clone the repository:**
```bash
git clone <repository-url>
cd "Challenge #1"
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Kaggle API Setup

To download the dataset, you need to configure the Kaggle API:

1. **Get your Kaggle API token:**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to the **API** section
   - Click **Create New API Token**
   - Download the `kaggle.json` file

2. **Configure the API token:**

   **On Windows (Git Bash/PowerShell):**
   ```bash
   mkdir -p ~/.kaggle
   cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **On macOS/Linux:**
   ```bash
   mkdir -p ~/.kaggle
   cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Alternative: Place directly in project:**
   ```bash
   mkdir kaggle
   cp /path/to/your/kaggle.json ./kaggle/kaggle.json
   export KAGGLE_CONFIG_DIR=./kaggle
   ```

### 4. Dataset Download

**Option 1: Using the provided script**
```bash
cd data
bash download.sh
cd ..
```

**Option 2: Manual download**
```bash
kaggle datasets download -d your-username/soil-classification-2025
unzip soil-classification-2025.zip -d data/
```

Expected dataset structure after download:
```
data/
└── data/
    └── soil_classification-2025/
        ├── train/
        ├── └──[train images] 
        ├── sample_submission.csv
        ├── test_ids.csv
        ├── train_labels.csv
        └── test/
            └── [test images]
```

## Usage

### Training the Model

1. **Open the training notebook:**
   ```bash
   jupyter notebook notebooks/training.ipynb
   ```

2. **Run all cells to:**
   - Load and preprocess the soil classification dataset
   - Initialize the TinyVGG model architecture
   - Train the model with proper validation
   - Evaluate performance and generate metrics
   - Save trained model weights

3. **Training outputs:**
   - Model weights: `notebooks/tinyvgg_soil_classifier.pth`
   - Metrics: `docs/cards/ml-metrics.json`
   - Architecture diagram: `docs/cards/tinyvgg_architecture.png`

### Running Inference

1. **Open the inference notebook:**
   ```bash
   jupyter notebook notebooks/inferencing.ipynb
   ```

2. **Run all cells to:**
   - Load the pre-trained model
   - Process test images
   - Generate soil type predictions
   - Create submission file

3. **Inference outputs:**
   - Predictions: `notebooks/submission.csv`

### Using Essential Functions

The `src/essentials.py` module contains utility functions:

```python
from src.essentials import load_data, preprocess_image, visualize_predictions

# Example usage
from essentials import SoilClassification, TinyVGG

# Instatinating SoilClassification class

data = SoilClassification(
    df = train_df,
    root_dir = soil_classification_path,
    transform = transform,
)
```

##  Model Architecture

The project uses a custom **TinyVGG** architecture specifically designed for soil classification:

- **Input:** RGB images (224x224)
- **Architecture:** Lightweight CNN with batch normalization and dropout
- **Output:** 4 soil type classes (Alluvial, Black, Clay, Red)
- **Optimization:** Adam optimizer

## Expected Results

After training, you should expect:
- **Training Accuracy:** ~85-95%
- **Validation Accuracy:** ~80-90%
- **Model Size:** 123 mb
- **Inference Time:** <3.10s per batch of 32

Detailed metrics are saved in `docs/cards/ml-metrics.json`.

##  Troubleshooting

### Common Issues

**1. Dataset not found error:**
```bash
# Verify dataset structure
ls -la data/data/soil_classification-2025/
# Should show train/ and test/ directories
```

**2. Kaggle API authentication error:**
```bash
# Check if kaggle.json is in the right location
ls -la ~/.kaggle/kaggle.json
# Verify permissions
ls -l ~/.kaggle/kaggle.json  # Should show -rw-------
```

**3. CUDA/GPU issues:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

**4. Module import errors:**
```bash
# Ensure you're in the project root directory
pwd  # Should end with "Challenge #1"
# Check if src/__init__.py exists
ls src/__init__.py
```

### Performance Optimization

- **For faster training:** Ensure CUDA is available and properly configured
- **For memory issues:** Reduce batch size in training notebook
- **For better accuracy:** Experiment with data augmentation parameters

##  File Descriptions

| File/Directory | Description |
|----------------|-------------|
| `notebooks/training.ipynb` | Complete model training pipeline |
| `notebooks/inferencing.ipynb` | Model inference and prediction generation |
| `src/essentials.py` | Core utility functions and helpers |
| `docs/cards/project-card.ipynb` | Model documentation and project card |
| `data/download.sh` | Automated dataset download script |
| `requirements.txt` | Python package dependencies |
