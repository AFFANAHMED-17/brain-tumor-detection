# 🧠 Brain Tumor Detection Using YOLO

An AI-powered computer vision project that uses the YOLO object detection framework to identify and localize brain tumor regions in medical images using bounding-box annotations.

The project explores deep learning, medical image analysis, object detection, GPU-accelerated model training, and model evaluation using a custom-labeled dataset.

> **Disclaimer:** This project is intended for educational and research purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical evaluation.

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Dataset Preparation](#-dataset-preparation)
- [Installation](#-installation)
- [GPU and CUDA Setup](#-gpu-and-cuda-setup)
- [Dataset Configuration](#-dataset-configuration)
- [Training the Model](#-training-the-model)
- [Model Evaluation](#-model-evaluation)
- [Making Predictions](#-making-predictions)
- [Understanding the Results](#-understanding-the-results)
- [Troubleshooting](#-troubleshooting)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Disclaimer](#-disclaimer)

## 🔍 Overview

Brain tumor detection is an important computer vision application in medical image analysis. Manually examining large numbers of medical images can be time-consuming, motivating research into automated image analysis systems.

This project applies **YOLO (You Only Look Once)**, a family of real-time object detection models, to detect and localize annotated tumor regions in images.

Unlike image classification, which predicts a class for an entire image, object detection predicts both the class and the location of objects using bounding boxes.

The model learns visual patterns from the training images and their corresponding annotations. Once trained, it can generate predictions on previously unseen images.

### Key Features

- Custom dataset configuration using a YOLO YAML file.
- Object detection using pretrained YOLO weights.
- Transfer learning on a labeled brain tumor dataset.
- GPU-accelerated training where supported.
- Validation using object detection metrics.
- Inference on individual images.
- Training output and experiment management.

## 🎯 Objectives

1. Understand the end-to-end workflow of an object detection project.
2. Prepare and configure an annotated image dataset for YOLO.
3. Fine-tune pretrained model weights on a domain-specific dataset.
4. Use GPU acceleration to improve training efficiency.
5. Evaluate detection performance using precision, recall, and mean Average Precision.
6. Identify dataset quality and annotation issues that affect model performance.
7. Build a reproducible foundation for further experimentation in medical computer vision.

## ⚙️ How It Works

The project follows this pipeline:

1. **Dataset preparation:** Organize brain tumor images and their corresponding bounding-box labels.
2. **Dataset configuration:** Define the training, validation, and test image paths and class names in `brain-tumor.yaml`.
3. **Transfer learning:** Initialize the detector using pretrained YOLO weights.
4. **Training:** Fine-tune the model on the custom dataset.
5. **Validation:** Evaluate the trained model on validation images.
6. **Inference:** Run the trained model on new images to generate predicted bounding boxes and confidence scores.

```text
Brain Tumor Image Dataset
          |
          v
Image and Annotation Preparation
          |
          v
brain-tumor.yaml
          |
          v
Pretrained YOLO Model
          |
          v
GPU-Accelerated Training
          |
          v
Validation and Evaluation
          |
          v
Trained Model Weights
          |
          v
Inference on New Images
          |
          v
Predicted Bounding Boxes
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO | Object detection and training framework |
| PyTorch | Deep learning backend |
| CUDA | GPU acceleration on supported NVIDIA hardware |
| OpenCV | Image processing and visualization |
| Jupyter Notebook | Experimentation and training workflow |
| YAML | Dataset configuration |
| NumPy | Numerical operations |

## 📁 Project Structure

The project directory is organized as follows:

```text
brain-tumor-detection/
│
├── brain-tumor.yaml       # Dataset configuration
├── train-note.ipynb       # Training and experimentation notebook
├── LICENSE.txt            # License information
├── .gitignore             # Files excluded from Git
├── README.md              # Project documentation
│
├── images/                # Dataset images, if stored here
├── labels/                # YOLO annotations, if stored here
│
└── runs/                  # Generated training and evaluation outputs
```

The exact dataset organization may differ depending on the paths defined in `brain-tumor.yaml`. Generated training outputs and downloaded model weights may be stored separately.

## 🗂️ Dataset Preparation

YOLO object detection requires images and matching annotation files.

### 1. Image data

The dataset contains images used to train and evaluate the detector. Images must be readable by the training framework and correctly associated with their annotation files.

### 2. YOLO annotation format

For object detection, each image typically has a corresponding `.txt` annotation file. Each annotated object is represented by a row containing:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized relative to the image dimensions and generally range from 0 to 1.

For example:

```text
0 0.52 0.48 0.24 0.30
```

This is an illustrative annotation, not a measured annotation from the project dataset.

### 3. Dataset splits

A typical dataset uses separate subsets:

- **Training set:** Used to update model parameters.
- **Validation set:** Used to monitor model performance during training and guide model selection.
- **Test set:** Used for final evaluation on data not used for training or model selection.

The actual split ratios should be based on the dataset and its configuration.

### Important dataset checks

Before training, verify that:

- Each image has the correct annotation file.
- Bounding-box coordinates are valid.
- Class IDs match the configured class names.
- Images and labels use matching filenames.
- Training and validation sets contain representative examples.
- Duplicate or near-duplicate images do not cause data leakage.
- The annotations accurately describe the visible tumor regions.

Dataset quality is critical: incorrect labels, inconsistent bounding boxes, class imbalance, and insufficient image diversity can substantially reduce detection performance.

## 💻 Installation

### Prerequisites

- Windows 10 or Windows 11.
- Python compatible with the selected PyTorch and Ultralytics versions.
- Git (optional, for version control).
- An NVIDIA GPU for CUDA-accelerated training, if available.
- A properly configured dataset.

### 1. Clone the repository

After publishing the repository to GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run the environment's Python executable directly.

### 3. Install the dependencies

For the Ultralytics YOLO workflow:

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python jupyter
```

Install a compatible PyTorch build separately if needed, following the official PyTorch installation instructions.

Official resources:

- [Ultralytics documentation](https://docs.ultralytics.com/)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics)

For reproducible experiments, record the package versions used for the successful training run.

## 🚀 GPU and CUDA Setup

GPU acceleration can substantially reduce training time compared with CPU-only training, depending on the model, dataset, and hardware.

### 1. Check NVIDIA GPU availability

Run:

```bash
nvidia-smi
```

If the command works, it displays information about the NVIDIA GPU and installed driver.

### 2. Check PyTorch CUDA support

Run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is available, the training command can explicitly select the first GPU using `device=0`.

If CUDA is unavailable, check the NVIDIA driver, the installed PyTorch build, and the compatibility requirements. Installing the standalone CUDA Toolkit alone does not guarantee that PyTorch can access the GPU.

## 📝 Dataset Configuration

The project includes `brain-tumor.yaml`, which specifies the dataset location and class information.

A typical Ultralytics detection dataset configuration looks like this:

```yaml
path: ./dataset

train: images/train
val: images/val
test: images/test

names:
  0: brain_tumor
```

This is an example configuration only. The paths and class names must match the actual dataset.

If your dataset contains multiple classes, define each class with the correct numeric ID.

For example, a configuration containing two classes would use IDs `0` and `1`, with names that match the annotations.

Before starting training, inspect your existing `brain-tumor.yaml` and make sure that all paths resolve correctly.

## 🧠 Training the Model

The project uses a pretrained YOLO model as the starting point for fine-tuning.

The project directory contains `yolo26n.pt`. Confirm that this checkpoint is compatible with the installed Ultralytics version before training.

### 1. Open the training notebook

Open the project notebook:

```bash
jupyter notebook train-note.ipynb
```

You can also open `train-note.ipynb` directly in Visual Studio Code.

### 2. Understand transfer learning

Instead of training a detector from scratch, transfer learning starts with weights learned from an earlier training task.

Fine-tuning adapts those learned features to the target dataset. This can reduce the amount of training required, although the final performance depends on the pretrained model, dataset quality, and training configuration.

### 3. Train using the command line

An example training command for a compatible Ultralytics YOLO model is:

```bash
yolo detect train model=yolo26n.pt data=brain-tumor.yaml epochs=100 imgsz=640 batch=16 device=0
```

Parameters:

| Parameter | Meaning |
|---|---|
| `model` | Pretrained checkpoint used to initialize training |
| `data` | Path to the dataset YAML file |
| `epochs` | Maximum number of training epochs |
| `imgsz` | Image size used for training |
| `batch` | Number of images processed per batch |
| `device` | `0` selects the first CUDA GPU |

These are example settings, not verified records of the parameters used in the original training run. Adjust the batch size to fit available GPU memory. If the GPU does not have enough memory, try a smaller batch size.

If the installed framework does not recognize the `yolo26n.pt` checkpoint, verify the supported model version and obtain the appropriate compatible weights before continuing.

### 4. Train using Python

The equivalent general Python workflow is:

```python
from ultralytics import YOLO

# Load a compatible pretrained detection model
model = YOLO("yolo26n.pt")

# Fine-tune on the configured dataset
results = model.train(
    data="brain-tumor.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

Run this code from the project root so the relative paths resolve correctly.

Use `device="cpu"` for CPU-only training, or `device=0` when CUDA is available.

### 5. Training outputs

Ultralytics normally creates a run directory under `runs/detect/` containing outputs such as:

- Training and validation metrics.
- Loss curves and metric plots.
- Prediction visualizations.
- Model checkpoints, including `best.pt` and `last.pt`, when saved by the training run.

The exact files depend on the framework version and training options.

`best.pt` typically represents the checkpoint selected by the framework's validation fitness criterion, while `last.pt` represents the most recent saved checkpoint.

Keep important experiment results locally and document the final metrics you actually obtain.

## 📊 Model Evaluation

A detection model should be evaluated on data that was not used to update its weights.

Common object detection metrics include:

### Precision

Precision measures how many predicted detections are correct.

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

Here, `TP` represents true positives and `FP` represents false positives.

### Recall

Recall measures how many ground-truth objects are detected.

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

Here, `FN` represents false negatives.

### Intersection over Union (IoU)

IoU measures the overlap between a predicted bounding box and a ground-truth bounding box.

\[
IoU=\frac{\text{Area of Intersection}}{\text{Area of Union}}
\]

### Mean Average Precision (mAP)

- **mAP@0.5:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP@0.5:0.95:** Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 in increments of 0.05.

For a single-class detector, these metrics evaluate detection quality for that class. The appropriate interpretation depends on the dataset, evaluation settings, and class definitions.

### Run validation

After training, validate the selected checkpoint:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=brain-tumor.yaml imgsz=640 device=0
```

Replace the checkpoint path with the actual path created by your training run.

**Do not report accuracy or mAP values until they have been measured on the actual evaluation data.** High validation metrics alone do not establish clinical reliability.

## 🔎 Making Predictions

After training, use the selected checkpoint to run inference on an image.

### Command-line inference

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg conf=0.25 save=True
```

Replace `path/to/image.jpg` with the path to an actual image.

The `conf` parameter sets the confidence threshold for retaining predictions. The example threshold is adjustable and should not be interpreted as a clinically appropriate cutoff.

### Python inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    conf=0.25,
    save=True
)

for result in results:
    print(result.boxes)
```

The prediction results can contain bounding-box coordinates, confidence scores, and predicted class IDs.

By default, saved prediction visualizations are placed in an Ultralytics output directory, typically under `runs/detect/predict/`.

## 📈 Understanding the Results

Evaluate the model using more than a single metric.

- Inspect predicted bounding boxes against the original annotations.
- Review false positives and false negatives.
- Check whether small or low-contrast tumor regions are consistently missed.
- Compare training and validation losses to identify possible overfitting.
- Inspect performance across different image characteristics and relevant subgroups.
- Evaluate the final model on a separate test set, if available.

If the detector produces many incorrect boxes or misses true tumor regions, possible causes include poor-quality annotations, insufficient training examples, class imbalance, domain differences, or unsuitable training settings.

Changing the architecture alone may not resolve these problems. Dataset inspection and error analysis should come first.

## ⚠️ Troubleshooting

| Problem | What to check |
|---|---|
| CUDA is unavailable | Verify the NVIDIA driver and CUDA-enabled PyTorch installation |
| CUDA out-of-memory error | Reduce batch size or image size |
| Dataset path not found | Check the paths in `brain-tumor.yaml` |
| Missing labels | Check annotation filenames and dataset structure |
| Unsupported checkpoint | Verify compatibility between the weights and Ultralytics version |
| Poor detection results | Inspect annotation quality, dataset diversity, class distribution, and validation errors |
| Training is slow | Confirm GPU utilization and review batch size and model complexity |
| No predictions appear | Verify the checkpoint, input image, confidence threshold, and class configuration |

## 🔬 Limitations and Future Improvements

### Current limitations

- Performance depends heavily on dataset size, annotation quality, and image diversity.
- A model trained on one dataset may not generalize to images from different sources or acquisition settings.
- False positives and false negatives remain possible.
- Detection performance does not establish diagnostic accuracy or clinical safety.
- Training and evaluation metrics must be verified on representative, independent data.

### Potential future improvements

- Perform systematic annotation quality checks.
- Evaluate generalization using an independent test dataset.
- Compare different YOLO model sizes and training configurations.
- Tune confidence and IoU thresholds using validation data.
- Analyze per-class metrics and difficult examples.
- Track experiments and compare model checkpoints.
- Develop a simple inference interface for research demonstrations.
- Investigate explainability, robustness, and domain adaptation.
- Consider appropriate clinical validation and expert review before any medical use.

## 📚 References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## 👨‍💻 Author

**Affan Ahmed**

Computer Science and Engineering | AI/ML | Computer Vision

- GitHub: [@AFFANAHMED-17](https://github.com/AFFANAHMED-17)
- LinkedIn: [Affan Ahmed](https://www.linkedin.com/in/affan-ahmed-athamsat/)

---

If you find this project useful for learning about object detection and computer vision, feel free to explore the code and experiment with the training workflow.

**Note:** Update the dataset details, model configuration, training parameters, evaluation results, and license information to reflect the actual experiment before publishing the repository.
