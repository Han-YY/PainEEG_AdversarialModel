# Adversarial Model for Pain Assessment with EEG
(The codes in Tensorflow have some issues, only the pytorch ones can work successfully)

# 1 Introduction
## 1.1 Background
Based on my previous work [1], the phase-based connectivity extracted from alpha band (4-8 Hz) can be utilized as ideal features input into machine learning model to classify the tonic pain-related conditions and non-pain conditions from EEG signals. However, it is still a big challenge to predict the conditions of a new particpant whose data was not involved in the training set, which made it very difficult to apply such models to unresponsive patients. 

The work in [2, 3] inspired me to use the adversarial model learning the features related to individual differences in an adversary network and increase the prediction accuracy of the main classifier, this project is for implementing such a target.

## 1.2 Feature
The EEG signals were preprocessed in a pilot progress and the features were extracted and saved in .npz files.
### 1.2.1 Inter-site phase clustering (ISPC)
The connectivity features were calculated as ISPCs:
$ ISPC_{xy}=|\frac{1}{n}\sum_{t=1}^{n}e^{i[\phi_x(t)-\phi_y(t)]}|$
where x and y are channels x and y, $\phi_x(t)$ means the phase of channel x at time point t.

### 1.2.2 Size of features
The features were reorganized as square matrix, whose size is (sample, channel, channel, 1)

## 2 Model
All the models and related classes are in trans_net.
### 2.1 Encoder (trans_net.encoder())
The encoder is shared by the main network and the adversary network, which aims to optimize the input.

### 2.2 Main classifier (trans_net.main_clf())
The classifier to classify the pain-related conditions, whose loss is minimized in the adversaral training progress.

### 2.3 Adversary classifier (trans_net.adv_clf())
The adversary classifier to classify the participant ids, whose loss is minimized in the pre-training progress and minimzed in the adversarial training.

### 2.4 PainDataset (trans_net.PainDataset)
- data_samples: connectivity features
- class_labels: list of condition labels
- subject_labels: list of subject ids

# 3 Framework
- exclude_id: the excluded participant's id for testing the transfer learning performance
- lam: lambda for controling the weights of the adversary network's loss

Applying adversarial model into the model predicting pain with EEG for transfer learning
Some codes were adapted from https://github.com/philipph77/ACSE-Framework and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# References:
[1] Han, Y., Valentini, E., & Halder, S. (2022, July). Classification of Tonic Pain Experience based on Phase Connectivity in the Alpha Frequency Band of the Electroencephalogram using Convolutional Neural Networks. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 3542-3545). IEEE.

[2] Bethge, D., Hallgarten, P., Özdenizci, O., Mikut, R., Schmidt, A., & Grosse-Puppendahl, T. (2022). Exploiting Multiple EEG Data Domains with Adversarial Learning. arXiv preprint arXiv:2204.07777.

[3] Smedemark-Margulies, N., Wang, Y., Koike-Akino, T., & Erdogmus, D. (2022, July). AutoTransfer: Subject transfer learning with censored representations on biosignals data. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 3159-3165). IEEE.

