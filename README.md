# sibnet
## Saliency Inductive Bias Net

### This is the official repository for SIBNet, Saliency Inductive Bias Net. 

From study:
Interpretability-Guided Inductive Bias For Deep Learning Based Medical Image Classification And Segmentation
Authors: Dwarikanath Mahapatra, Alexander Poellinger, Mauricio Reyes
Published at the Medical Image Analysis Journal.
[https://doi.org/10.1016/j.media.2022.102551][https://doi.org/10.1016/j.media.2022.102551]

Abstract: Deep learning methods provide state of the art performance for supervised learning
based medical image analysis. However it is essential that trained models extract
clinically relevant features for downstream tasks as, otherwise, shortcut learning and
generalization issues can occur. Furthermore in the medical field, trustability and transparency
of current deep learning systems is a much desired property. In this paper we
propose an interpretability-guided inductive bias approach enforcing that learned features
yield more distinctive and spatially consistent saliency maps for different class
labels of trained models, leading to improved model performance. We achieve our
objectives by incorporating a class-distinctiveness loss and a spatial-consistency regularization
loss term. Experimental results for medical image classification and segmentation
tasks show our proposed approach outperforms conventional methods, while
yielding saliency maps in higher agreement with clinical experts. Additionally, we show
how information from unlabeled images can be used to further boost performance. In
summary, the proposed approach is modular, applicable to existing network architectures
used for medical imaging applications, and yields improved learning rates, model
robustness, and model interpretability.

This repository includes the basic fundamental elements of SIBNet to get reseaerchers started into using it. 
![SIBNet AUC Benchmarking](/SIBNet_AUC.png)
![SIBNet Enhanced saliencymaps](/SIBNet_Saliencies.png)
