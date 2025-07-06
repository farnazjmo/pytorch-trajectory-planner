# MLP-Based Vehicle Trajectory Prediction

This project implements an MLP (Multilayer Perceptron) to predict future vehicle waypoints in a simplified driving simulation environment. It demonstrates a complete machine learning pipeline including model design, training, evaluation, and logging.

## 🔧 Features

- Implements a custom PyTorch MLP for trajectory regression
- Supports additional models: CNN and Transformer
- Full training loop with:
  - Smooth L1 loss
  - Metric logging (TensorBoard)
  - Model saving
- Modular and extensible for other input types (e.g., images)

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy
- TensorBoard

## 🚀 Usage

### ⚙️ Training the Model

To train the CNN-based planner:

```python
from train import train

train(
    model_name="cnn_planner",
    transform_pipeline="default",
    num_workers=4,
    lr=1e-2,
    batch_size=256,
    num_epoch=50,
)
```

### 🖥️ CLI Training

You can also run training directly from the terminal:
```bash
python train.py --model_name mlp_planner --num_epoch 40 --batch_size 64
```

## 🎥 Model Demos
###  MLP Planner

![MLP Demo](videos/mlp_planner_lighthouse.gif)

### CNN Planner

![CNN Demo](videos/cnn_planner_lighthouse.gif)

### Transformer Planner

![Transformer Demo](videos/transformer_planner_lighthouse.gif)


## 📄 Disclaimer
*This project shares only original code written by the author. No university-provided materials, datasets, or assignment instructions are included.*
