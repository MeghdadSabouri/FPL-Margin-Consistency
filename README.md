# Perterbation Fedelity Loss Loss (PFL) Margin Consistency for Lung Cancer Histology Classification
This repository presents a deep learning framework designed for classifying lung adenocarcinoma histologic patterns from whole-slide images (WSIs). The framework introduces a novel loss function—Perterbation Fedelity Loss Loss (PFL)—which leverages margin consistency concepts to improve robustness. The implementation combines an attention-based ResNet model with a custom loss function that integrates cross entropy loss, contrastive loss, and a perturbation fidelity loss. In addition, hyperparameter optimization using Bayesian methods is provided to fine-tune key parameters.
# Datasets and References
Dartmouth Lung Cancer Histology Dataset:
This dataset comprises 143 hematoxylin and eosin (H&E)-stained formalin-fixed paraffin-embedded (FFPE) whole-slide images of lung adenocarcinoma from the Department of Pathology and Laboratory Medicine at Dartmouth-Hitchcock Medical Center (DHMC). The images are de-identified and released with permission from the Dartmouth-Hitchcock Health (D-HH) Institutional Review Board (IRB). For further details, please refer to: "https://bmirds.github.io/LungCancer/"
# Acknowledgments
We would like to acknowledge and thank the respected scientists and open-source communities whose codes and ideas have significantly contributed to this work.

The foundational ideas and implementation details presented in the paper "Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers" by Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, and Christian Gagné have been instrumental in shaping our approach to robust classification and margin consistency.

We have cited and provided credit to the following resource for further reference:

https://arxiv.org/abs/2406.18451v2
# Overview
The project is organized into the following key components:

Model Definition 

Loss Function :
    Implements the custom Perterbation Fedelity Loss Loss (PFL) which is designed to promote margin consistency. This loss is a combination of:
        Cross Entropy Loss
        Contrastive Loss
        Perturbation Fidelity Loss: Encourages robustness by integrating perturbation-based fidelity measures inspired by margin consistency principles.
        
Training Pipeline 
        Load and preprocess WSIs using a custom dataset loader.
        Train and validate the model while tracking key performance metrics (loss, overall accuracy, and per-class accuracy).
        Visualize training progress through detailed plots.
        
Hyperparameter Optimization 
    Uses Bayesian optimization to fine-tune key hyperparameters such as the structure preservation strength (alpha), the random perturbation magnitude (beta), and the temperature parameter for contrastive loss.
# Contact
If you have any questions or comments, please feel free to contact the author, Meghdad Sabouri Rad, at Meghdad.Sabouri.Rad@gmail.com.
