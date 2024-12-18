# Auto Deep Learning (AutoDL) Projects

**Auto Deep Learning (AutoDL)** is an open-source, lightweight but powerful framework for automating deep learning tasks like neural architecture search (NAS) and hyperparameter optimization (HPO). This project aims to provide a simple yet effective library that allows researchers, engineers, and beginners to easily test and use various state-of-the-art NAS and HPO algorithms.

## Table of Contents

- [Introduction](#introduction)
- [Who Should Use AutoDL](#who-should-use-autodl)
- [Why Use AutoDL](#why-use-autodl)
- [AutoDL Capabilities](#autodl-capabilities)
- [Installation](#installation)
- [Training and Usage](#training-and-usage)
- [Citation](#citation)
- [License](#license)
- [Contribution](#contribution)
- [Additional Notes](#additional-notes)

## Introduction

AutoDL Projects implements various cutting-edge Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) algorithms to make the deployment of deep learning models easier and more efficient. These algorithms help automate model architecture design and hyperparameter tuning, significantly improving the accuracy and efficiency of machine learning models.

This repository contains popular algorithms, including:

- **DARTS**: Differentiable Architecture Search
- **GDAS**: Robust Neural Architecture Search
- **TAS**: Transformable Architecture Search
- **SETN**: Self-Evaluated Template Network
- **NAS-Bench-201**: Benchmark for NAS
- **NATS-Bench**: Benchmarking NAS Algorithms

Additionally, there is support for basic models like **ResNet** for image classification and hyperparameter optimization via methods like **HPO-CG**.

## Who Should Use AutoDL

- **Beginners**: New to deep learning and want to try different NAS and HPO algorithms without dealing with complex details.
- **Engineers**: Professionals aiming to automate the process of model selection and hyperparameter tuning for real-world applications.
- **Researchers**: Academics looking for a simple yet flexible framework to test and benchmark new NAS and HPO algorithms.

## Why Use AutoDL

- **User-Friendly**: Simplifies the process of model building by automating architecture selection and hyperparameter optimization.
- **Open Source**: Free and open for modification and redistribution under the MIT license.
- **Active Maintenance**: Constant updates and new algorithms are added to the codebase.
- **Efficient**: Automates repetitive tasks like hyperparameter tuning and architecture search, saving time and resources.

## AutoDL Capabilities

This project offers a variety of algorithms and scripts for Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO):

### NAS Algorithms

| Type | ABBRV | Algorithm | Description |
|------|-------|-----------|-------------|
| **NAS** | TAS | Network Pruning via Transformable Architecture Search | Prunes architectures through NAS (NeurIPS-2019). |
| **NAS** | DARTS | Differentiable Architecture Search | Differentiable NAS for efficient architecture search (ICLR-2019). |
| **NAS** | GDAS | Robust Neural Architecture Search | Optimized for robustness and computational efficiency (CVPR-2019). |
| **NAS** | SETN | One-Shot Neural Architecture Search via Self-Evaluated Template Network | Self-evaluated template network for one-shot NAS (ICCV-2019). |
| **NAS** | NAS-Bench-201 | NAS-Bench-201: Extending the Scope of Reproducible NAS | A benchmark for reproducible NAS experiments. |
| **NAS** | NATS-Bench | Benchmarking NAS Algorithms for Architecture Topology and Size | Comprehensive benchmark for NAS algorithms. |
| **NAS** | ENAS/REA/REINFORCE | Various additional algorithms | Check relevant papers for more details. |

### HPO Algorithms

| Type | ABBRV | Algorithm | Description |
|------|-------|-----------|-------------|
| **HPO** | HPO-CG | Hyperparameter Optimization with Approximate Gradient | Hyperparameter tuning with gradient approximation (coming soon). |

### Basic Models

| Type | Model | Description |
|------|-------|-------------|
| **Basic** | ResNet | Deep learning-based image classification using ResNet. |

## Installation

### Prerequisites
- **Python**: 3.6 or higher
- **PyTorch**: 1.5.0 or higher
- **Dependencies**: `numpy`, `scikit-learn`, `opencv`, `matplotlib`

### Steps for Installation:

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/AutoDL-Projects.git
   cd AutoDL-Projects
   pip install -r requirements.txt
###Training and Usage
##Training a Model:
To train a model using your dataset, use the following command:


    python train_model.py --data <path_to_data>
Where:

--data: Path to the dataset.
Hyperparameter Optimization:
To perform hyperparameter optimization:


    python optimize_hyperparameters.py --data <path_to_data> --config <config_file>
#Where:

--data: Path to your dataset.
--config: Path to a configuration file specifying hyperparameter optimization parameters.
###Model Prediction:
To make predictions with a trained model:

Image-Based Prediction:

    python predict_helmet_violation.py --image <path_to_image>
Video-Based Prediction:


    python predict_helmet_violation.py --video <path_to_video>
###Command-Line Arguments:
  --image: Path to an image file for prediction.
--video: Path to a video file for prediction.
###Citation
If you find this project helpful, please cite the related papers:

b

###License
This project is licensed under the MIT License. See the LICENSE file for more details.

##Contribution
We welcome contributions to the project! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes.
Submit a pull request describing the changes.
Please refer to the CONTRIBUTING.md file for guidelines.
