# Soil Classification using Vision Transformers

This project addresses a binary classification task: identifying whether an image contains soil or not. The solution is based on pretrained Vision Transformers (ViT) and a lightweight TinyVGG model. Classification is performed using cosine similarity between embeddings and thresholding.

## Folder Structure

```
Challenge #2/
├── src/
│   └── essentials.py            # Contains Class definations
├── models/
│   ├── tinyvgg_model_state_dict.ipynb   # TinyVGG training and feature extraction
│   └── vit_model_state_dict.ipynb       # ViT evaluation and visualization
├── docs/
│   ├── architecture.png         # General model architecture diagram
│   ├── 2_tinyvgg.png            # TinyVGG architecture visualization
│   ├── 2_vit.png                # ViT architecture visualization
│   └── cards/
│       ├── project-card.ipynb   # Notebook for evaluation and reporting
│       └── ml-metrics.json      # Evaluation metrics
├── data/
│   └── download.sh
├── requirements.txt          # Dataset download instructions (optional)
└── README.md
```

## Problem Statement

The goal is to build a classifier that determines whether a given image contains soil. Only positive class (soil) embeddings are used as a reference. Test embeddings are compared to these references, and predictions are made based on a distance threshold.

### Key Challenges
- Binary classification with limited positive data
- Threshold optimization for cosine similarity
- Embedding-based classification approach

## Approach

1. **Feature Extraction:** Extract embeddings using a pretrained ViT or TinyVGG model
2. **Similarity Computation:** Compute cosine similarity between test embeddings and reference (soil) embeddings
3. **Threshold Classification:** Classify based on a chosen threshold value
4. **Model Comparison:** Compare results across both models to evaluate performance differences

### Classification Pipeline
```
Input Image → Feature Extraction → Embedding Vector → Cosine Similarity → Threshold → Prediction
```

## Model Architectures

### Vision Transformer (ViT)
- **Base Model:** ViT-Base pretrained on ImageNet-1k
- **Input Size:** 224x224 pixels
- **Embedding Dimension:** 1000
- **Advantages:** Strong feature representation, good generalization

### TinyVGG
- **Architecture:** Lightweight CNN with 2 convolutional blocks
- **Input Size:** 224x224 pixels  
- **Embedding Dimension:** 33160
- **Advantages:** Fast inference, lower computational requirements

Architecture visualizations are provided in `docs/architecture.png`, `docs/2_tinyvgg.png`, and `docs/2_vit.png`.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Challenge #2"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

If a script is provided:
```bash
cd data
bash download.sh
```
Or manually place the dataset as required by the notebooks.

---

## Usage

All evaluation and analysis is performed in the notebook:

1. **Open the evaluation notebook**

   - Open `docs/cards/project-card.ipynb` in Jupyter or VS Code.

2. **Run all cells**

   - The notebook will:
     - Load the pretrained model weights from `/models/`
     - Load the dataset from `/data/`
     - Perform evaluation and show results/metrics

**Note:**  
There are no standalone training or inference scripts. All evaluation is performed using the provided notebook and pretrained weights.

---

## Evaluation Metrics

Current model performance metrics are stored in `/docs/cards/ml-metrics.json`:

```json
{
  "model_comparison": {
    "ViT": {
      "f1_score": 1.0000,
      "optimal_threshold": 0.1
    },
    "TinyVGG": {
      "min_f1_score": 0.9823,
      "optimal_threshold": 0.001
    }
  }
}
```

### Performance Summary
- **Best Model:** Vision Transformer (ViT)
- **F1 Score:** 1.0
- **Accuracy:** 98%
- **Optimal Threshold:** 0.1

---

## Model Cards

### ViT Model Card
```yaml
Model Name: Soil Classification ViT
Version: 1.0
Architecture: Vision Transformer (ViT-Base)
Training Data: Multi Class Soil Dataset from the competition
Performance:
  - Accuracy: 98%
  - Macro F1 Score: 0.9892
  - Weighted F1 Score: 0.9877
  - Macro Precision: 0.9904
  - Macro Recall: 0.9881
  - Per-Class F1 Scores:
      Class 0: 0.9955
      Class 1: 0.9714
      Class 2: 1.0000
      Class 3: 1.0000
Confusion Matrix:
  - [[102, 1, 0, 0],
     [2, 51, 0, 0],
     [0, 0, 41, 0],
     [0, 0, 0, 48]]
Limitations:
  - Requires high-resolution images for best performance
  - Computationally intensive compared to CNN alternatives
Use Cases:
  - Agricultural soil analysis
  - Construction site monitoring
  - Environmental research
```

### TinyVGG Model Card
```yaml
Model Name: Soil Classification TinyVGG
Version: 1.0
Architecture: Lightweight CNN (2 Conv blocks)
Training Data: Multi Class Soil Dataset from the competition
Performance:
  - Accuracy: 97.14%
  - Macro F1 Score: 0.9771
  - Weighted F1 Score: 0.9797
  - Macro Precision: 0.9742
  - Macro Recall: 0.9803
  - Per-Class F1 Scores:
      Class 0: 0.9796
      Class 1: 0.9892
      Class 2: 0.9474
      Class 3: 0.9920
Confusion Matrix:
  - [[96, 3, 0, 0],
     [0, 46, 1, 0],
     [1, 0, 36, 0],
     [0, 0, 0, 62]]
  - Note: This model was only trained on jpeg type images
Limitations:
  - Slightly lower accuracy than ViT, especially for Class 2
  - May struggle with complex soil textures
Use Cases:
  - Real-time mobile applications
  - Edge deployment scenarios
  - Resource-constrained environments
```

---

## Results and Analysis

### Key Findings
1. **ViT outperforms TinyVGG** in all metrics but requires more computational resources.
2. **Optimal thresholds differ** between models (0.1 for ViT, 0.001 for TinyVGG).
3. **Cosine similarity approach** works well for this embedding-based classification.
4. **Minimum F1 scores** (ViT: 1.000, TinyVGG: 0.9825) show ViT is better.


