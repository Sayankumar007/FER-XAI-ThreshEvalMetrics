# FER-XAI-ThreshEvalMetrics

## Revealing the Human-like Similarities in Automated Facial Expression Recognition:  
### An Empirical Investigation using eXplainable Artificial Intelligence

---

## Authors

**Sayan Kumar Bhowmick**  
Department of Computer Science and Engineering  
Jalpaiguri Government Engineering College, West Bengal, India  

**Asit Barman***†  
Department of Information Technology  
Siliguri Institute of Technology, West Bengal, India  
📧 mtechitasit@gmail.com  

**Swalpa Kumar Roy**†  
Department of Computer Science and Engineering  
Alipurduar Government Engineering and Management College, West Bengal, India  

**Paramartha Dutta**†  
Department of Computer and System Sciences  
Visva-Bharati University, Santiniketan, West Bengal, India  

\* Corresponding Author  
† These authors contributed equally to this work.

---

## Abstract

Human behavior analysis significantly depends on facial expression recognition, where deep learning has enabled the development of models capable of achieving human-level performance. Explainable Artificial Intelligence (XAI) techniques are employed to validate the trustworthiness of trained convolutional neural networks by generating interpretable saliency heatmaps using methods such as GradCAM, GradCAM++, LayerCAM, and ScoreCAM. These heatmaps highlight critical facial regions utilized by the classifiers, thereby aligning system behavior with human cognitive processes.

Existing evaluation metrics, including average drop, confidence increase, and win percentage, assess system reliability but fail to quantify trustworthiness. This work introduces **Thresholding-based Evaluation Metrics**, formulated in terms of Precision, Recall, and F-measure, to both assess and quantify the reliability of XAI techniques for a given classifier. Experiments are conducted on three benchmark datasets—CK+, RAFD, and RAF-DB—using multiple deep learning architectures, including VGG19, ResNet18, GoogLeNet, DenseNet121, and EfficientNet. The results demonstrate that the proposed metrics are as effective as traditional metrics while providing an explicit quantitative measure of trustworthiness, enhancing their suitability for human-centered and ethical AI applications.

---

## Repository Overview

This repository provides the implementation and experimental framework associated with the above-mentioned research work. It supports the reproducibility of the empirical findings by including:

- Training and evaluation of deep CNN-based FER models  
- Generation of saliency maps using multiple XAI techniques  
- Implementation of proposed Thresholding-based Evaluation Metrics  
- Comparative analysis across datasets, models, and explainability methods  

---

## Datasets

The experiments are conducted on the following benchmark facial expression recognition datasets:

- **CK+ (Extended Cohn-Kanade Dataset)**
- **RAFD (Radboud Faces Database)**
- **RAF-DB (Real-world Affective Faces Database)**

> Note: Dataset access must be obtained from the respective official sources.

---

## Models and XAI Techniques

### Deep Learning Models
- VGG19  
- ResNet18  
- GoogLeNet  
- DenseNet121  
- EfficientNet  

### Explainable AI Techniques
- GradCAM  
- GradCAM++  
- LayerCAM  
- ScoreCAM  

---

## Thresholding-based Evaluation Metrics

The proposed evaluation framework introduces:
- **Precision**
- **Recall**
- **F-measure**

These metrics quantify the reliability and trustworthiness of saliency-based explanations by thresholding activation maps and comparing them with facial region relevance.
