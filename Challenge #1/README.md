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

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Challenge #1"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Additional requirements (if not in requirements.txt):
   - torch
   - torchvision
   - torchview
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - tqdm
   - Pillow

3. **Download the dataset:**
   - If a script is provided:
     ```bash
     cd data
     bash download.sh
     ```
   - Or manually place the dataset in `data/data/soil_classification-2025/` as required by the notebooks.

4. **(Optional) For model architecture visualization:**
   - Install [Graphviz](https://graphviz.gitlab.io/download/) and add its `bin` directory to your system PATH.

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