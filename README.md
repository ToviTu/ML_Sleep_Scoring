# A Machine Learning-Based Sleep-Wake Analysis of Mouse Model of Alzheimer's Disease

## Introduction

Alzheimer’s Disease is an irreversible, neurodegenerative disorder. The molecular underpinnings of AD accumulate for decades before diagnoses are made, rendering treatments extremely difficult. Thus, improving current treatments requires a better understanding of the behavioral indicators of disease onset. The bi-directional relationship between sleep deficiency and disease progression allows for quantifying future disease risk. However, the conventional method of evaluating sleeping conditions requires extensive labor and has low timescale resolution. This work aims to build and evaluate the performance of ML models that learn from human experiences in sleep/wake scoring and analyze the sleep disorders trend in a mouse neurodegenerative disease model. We hypothesize that the predicted results support increased wakefulness in the ApoE4/tau mouse group, which correlates with age. Recordings of brain activity from 11 ApoE4/tau mice and 28 wild-type mice were used to train gradient-boosting trees and convolutional neural networks. Both models achieved an accuracy comparable to humans (≥ 85%). Statistical analysis of percentages of wakefulness in 24 hours supports that wakefulness is higher for TE4 mice since 200 days of age and has a positive polynomial correlation with age. A high level of wakefulness beyond a particular age point can predict the disease with high confidence. CNN models succeeded in upsampling the scoring labels to the milliseconds level and discovered transient neuropsychological events in wild-type animals that humans previously ignored. Future work involves investigating the presence of the new states in ApoE4/tau animals and their correlation with the disease.

## Methods

<img width="1016" alt="image" src="https://github.com/ToviTu/ML_Sleep_Scoring/assets/52998198/4772c587-faff-4776-b458-6b58079420d2">

(A) Recording of neural voltage data via a customized electrode probe implanted in the CA1 region of the hippocampal circuit.(1) (B) Significant prolonged time spent in Wake for TE4 animals calculated from human ground-truth (linear mixed effect model Percentage ~ Age + Genotype + (1| Animal)) p<0.05.  (C) A sample spectrogram with a range of 0.1-60Hz of local field potential (LFP) in 1 hour calculated from the neural recording data where wake, NREM, and REM are specified with colored lines in red, blue, and green, respectively.

## Models

<img width="484" alt="image" src="https://github.com/ToviTu/ML_Sleep_Scoring/assets/52998198/ef0df3ad-8237-4014-9c7a-887c5e5d3aec">

(A) (B) (C) Confusion matrices for XGBoost (2), gradient-boosting tree + K-nearest neighbors smoother, and convolutional neural network (CNN) (1) with different feature sets. (D)  CNN architecture with 7 interleaved convolutional layers and pooling layers trained on times-series data in 1s window and 64 channels.

## Results

<img width="991" alt="image" src="https://github.com/ToviTu/ML_Sleep_Scoring/assets/52998198/7fb0d779-0d7c-4191-8a33-e3d606cfbd4f">

(A) Comparison between the percentage time spent awake during dark and light hours (12 hours each) in a lab setting shows similar distribution for machine predictions and ground truth. (B) Mean percentage time spent in 3 sleep states using model prediction. The result is similar to human-scored data in Figure 1 (B).

<img width="1010" alt="image" src="https://github.com/ToviTu/ML_Sleep_Scoring/assets/52998198/bec50abc-d076-478e-b56c-8185afc52d98">

(C) Comparison of power distributions grouped by state labels between ground truth (upper 4) and model prediction (lower 4). (D) Regression analysis of the effect of age and genotype on the percentage of time in 24 hours spent awake. A linear mixed effect model (%Wake~Age+Genotype+(1|Animal)) supports a significant effect of Genotype with P=0.002.






