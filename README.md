# Revealing the Human-like Similarities in Automated Facial Expression Recognition  
## An Empirical Investigation using eXplainable Artificial Intelligence

## Authors

**Sayan Kumar Bhowmick**  
Department of Computer Science and Engineering  
Jalpaiguri Government Engineering College, West Bengal, India  
📧 sayankr.contact@gmail.com  

**Asit Barman**  
Department of Information Technology  
Siliguri Institute of Technology, West Bengal, India  
📧 mtechitasit@gmail.com  

**Swalpa Kumar Roy**  
Department of Computer Science and Engineering  
Alipurduar Government Engineering and Management College, West Bengal, India  

**Paramartha Dutta**  
Department of Computer and System Sciences  
Visva-Bharati University, Santiniketan, West Bengal, India  

---

## Affiliation
Developed under **Adaptive Intelligence LAB**  
An independent research group focused on AI & Machine Learning  

🔗 https://github.com/adaptive-intelligence-laboratory

---

## Abstract

Human behavior analysis significantly depends on facial expression recognition, where deep learning has enabled the development of models capable of achieving human-level performance. Explainable Artificial Intelligence (XAI) techniques are employed to validate the trustworthiness of trained convolutional neural networks by generating interpretable saliency heatmaps using methods such as GradCAM, GradCAM++, LayerCAM, and ScoreCAM. These heatmaps highlight critical facial regions utilized by the classifiers, thereby aligning system behavior with human cognitive processes.

Existing evaluation metrics, including average drop, confidence increase, and win percentage, assess system reliability but fail to quantify trustworthiness. This work introduces **Thresholding-based Evaluation Metrics**, formulated in terms of Precision, Recall, and F-measure, to both assess and quantify the reliability of XAI techniques for a given classifier. Experiments are conducted on three benchmark datasets—CK+, RAFD, and RAF-DB—using multiple deep learning architectures, including VGG19, ResNet18, GoogLeNet, DenseNet121, and EfficientNet. The results demonstrate that the proposed metrics are as effective as traditional metrics while providing an explicit quantitative measure of trustworthiness, enhancing their suitability for human-centered and ethical AI applications.

The source code will be made publicly available at:  
[https://github.com/Sayankumar007/FER-XAI-ThreshEvalMetrics](https://github.com/Sayankumar007/FER-XAI-ThreshEvalMetrics)


---

## Repository Overview

This repository provides the implementation and experimental framework associated with the above-mentioned research work. It supports the reproducibility of the empirical findings by including:

- Training and evaluation of deep CNN-based FER models  
- Generation of saliency maps using multiple XAI techniques  
- Implementation of proposed Thresholding-based Evaluation Metrics  
- Comparative analysis across datasets, models, and explainability methods  

---

## Dataset Structure

The code expects the dataset to be organized as follows:

```text
root_dir/
│
├── train/
│   ├── class_1/
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
│
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```
> **Important:** Each split (train/val/test) should contain at least one class folder with valid image files. The code includes an integrity check that will verify missing splits, empty folders, or corrupted images.


---
## Installation

Make sure Python 3.8+ is installed. Recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/Sayankumar007/FER-XAI-ThreshEvalMetrics.git
cd FER-XAI-ThreshEvalMetrics
```

## Environment & Reproducibility

This project has been developed and tested across multiple environments, including:

- Local workstation (Windows/Linux)
- Google Colab
- Kaggle Notebook environments

Due to differences in CUDA versions, PyTorch builds, and platform-specific configurations, minor version variations may exist across executions. However, the codebase is designed to remain compatible across PyTorch 2.x and standard scientific Python stacks.

### Python Version
Tested with:
- Python 3.8 – 3.10

### Dependency Installation

We provide two dependency files:

#### 1. General Compatibility Installation
For most users:

```bash
pip install -r requirements.txt
```

This installs platform-compatible versions of required packages.

#### 2. Exact Development Environment

For replicating the original development setup:

```bash
pip install -r requirements_exact.txt
```

This installs the exact package versions used during development.

### Notes on Reproducibility

Pretrained CNN backbones are obtained from torchvision. GPU execution depends on the CUDA version installed in the host system. Minor numerical differences may occur across hardware or CUDA versions. Random seeds should be fixed for deterministic experimentation where required. The dataset integrity check ensures structural consistency across environments.

Despite multi-platform testing, the experimental pipeline, evaluation metrics, and XAI visualizations demonstrate consistent qualitative and quantitative trends across environments.


---


## Usage

Run the `run.py` script to train/evaluate models:

```bash
python run.py <folder_name> <model_name> <root_dir> [--epochs N] [--lr LR] [--n N] [--visualize]
```

#### Arguments:

`folder_name` : Output folder to save `results`, PR curves, and visualizations

`model_name` : CNN model to run (vgg, resnet, googlenet, mobilenet, efficientnet, densenet)

`root_dir` : Root directory of the dataset (must include train, val, test subfolders)

#### Optional:

`--epochs` : Number of training epochs (default: 100)

`--lr` : Learning rate (default: 0.00001)

`--n` : Number of samples to visualize (default: 1)

`--visualize` : Save intermediate visualizations of saliency maps

Example:
```bash
python run.py results resnet ./RAF-DB --epochs 50 --lr 0.001 --n 5 --visualize
```
or simply
```bash
python run.py results resnet ./RAF-DB
```

## Citations

If you use this code or ideas in your research, please cite our work:

```bibTex
@article{bhowmick2026revealing,
  title={Revealing the human-like similarities in automated facial expression recognition: an empirical investigation using eXplainable artificial intelligence},
  author={Bhowmick, Sayan Kumar and Barman, Asit and Roy, Swalpa Kumar and Dutta, Paramartha},
  journal={Multimedia Tools and Applications},
  volume={85},
  number={3},
  pages={243},
  year={2026},
  publisher={Springer}
}
```

## Acknowledgements

Thanks to the dataset authors for providing CK+, RAFD, and RAF-DB datasets.
Thanks to PyTorch and the research community for open-source implementations of CNN models and XAI methods.
Special thanks to the mentors and colleagues who provided feedback during implementation.

## License

This repository is for research purposes only. Please refer to the LICENSE file for detailed terms.
