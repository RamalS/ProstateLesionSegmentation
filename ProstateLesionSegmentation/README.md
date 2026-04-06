# Prostate Lesion Segmentation Environment

## Description

Systems that enable fast and accurate detection and quantification of prostate tumor lesions are important in clinical practice, as they improve both the speed and quality of diagnosis. This project is part of a master’s thesis focused on developing and evaluating a method for the automatic segmentation of suspicious prostate lesions using deep neural networks.

The work includes analysis of prostate MRI modalities (T2-weighted, ADC, DWI), investigation of PI-RADS criteria, and exploration of modern segmentation approaches such as CNN- and transformer-based architectures. A multimodal model is implemented using one or more MRI sequences, including preprocessing steps like normalization and registration, followed by training for lesion segmentation.

Evaluation is performed using standard metrics such as Dice, IoU, Sensitivity, and Hausdorff distance, with comparisons across different model variants.

## Build

```bash
docker compose build
