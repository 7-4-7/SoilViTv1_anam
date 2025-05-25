# Soil Classification Challenge

This project is a deep learning solution for soil type classification using PyTorch. It includes data preprocessing, model training, evaluation, and inference pipelines. The model is based on a custom TinyVGG architecture.

## Project Structure

```
Challenge #1/
├── data/
|   └── download.sh                         # Download dataset
├── docs/
│   └── cards/
│       ├── ml-metrics.json                 # Metrics
│       └── tinyvgg_architecture.png
|       └──project-card.ipynb               # Contains model project card
├── notebooks/
│   ├── training.ipynb                      #Training Notebook
│   └── inferencing.ipynb                   #Testing Notebook
├── src/
|   └──__init__.py
│   └── essentials.py                        #Essential funtions
└── README.md
```

## Setup

NOTE : Please ensure you are working in a  venv before going through steps

0. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Challenge #1"
   ```

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. ## 2. Set up Kaggle API token

To download the dataset, you need a Kaggle API token:

1. Go to your Kaggle account: [https://www.kaggle.com/account](https://www.kaggle.com/account)  
2. Scroll to the **API** section and click **Create New API Token**  
3. This will download a file named `kaggle.json`

---

## 3. Place the Kaggle token and set permissions

### On Git Bash / Linux / macOS

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. **Download the dataset:**
   - If a script is provided:
     ```bash
     cd data
     bash download.sh
     ```
   - Or manually place the dataset in `data/data/soil_classification-2025/` as required by the notebooks.


## Usage

### Training

- Open `notebooks/training.ipynb` and run all cells to:
  - Load and preprocess data
  - Train the TinyVGG model
  - Evaluate and save metrics to `docs/cards/ml-metrics.json`
  - Visualize the model architecture with torchview

### Inference

- Open `notebooks/inferencing.ipynb` and run all cells to:
  - Load the trained model
  - Predict soil types for test images
  - Save predictions to `notebooks/submission.csv`

## Outputs

- **Model metrics:** `docs/cards/ml-metrics.json`
- **Model architecture diagram:** `docs/cards/tinyvgg_architecture.png`
- **Predictions:** `notebooks/submission.csv`

## License

[MIT License](LICENSE)
