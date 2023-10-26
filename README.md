# Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net

## I. INTRODUCTION

**Motivation:** The reason we're willing to put in the effort to do this is because through-the-wall radar (TWR) human activity recognition is indeed a worthwhile research topic, which can provide non-contact, real-time human movement monitoring, with a wide range of potential applications in areas such as urban safety and disaster rescue. Besides, Open-source efforts can foster collaboration and innovation in the field, advancing the technology and enabling our work to benefit the wider research community.

![image](https://github.com/JoeyBGofficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/8f4e46fc-40c3-4781-8737-da50f28e4a10)
Fig. 1. Prospects for the application of through-the-wall radar human monitoring.

**Basic Information:** This repository is the open source code for our latest feasibility work: "Abnormal Human Activity Recognition Method Based on Micro-Doppler Corner Representation and Non-Local Mechanisms for Through-the-Wall Radar", submitted to Journal of Radars;

**Submitted Author:** Weicheng Gao;

**Email:** JoeyBG@126.com;

**Abstract:** Through-the-wall radar is able to penetrate walls and achieve indoor human target detection. Deep learning is commonly used to extract the micro-Doppler signature, which can be used to effectively identify abnormal human activities behind obstacles. However, when different testers are invited to generate the training set and test set, the test accuracy of deep-learning-based recognition method is low with poor generalization ability. Therefore, this paper proposes an abnormal human activity recognition method based on micro-Doppler corner features and Non-Local mechanism. In this method, Harris and Moravec detectors are utilized to extract the corner features on the radar image, and the corner feature dataset is established in this manner. Then, multi-link parallel convolutions and Non-Local mechanism are utilized to construct the global contextual information extraction network to learn the global distribution characteristics of image pixels. The semantic feature maps are generated by repeating the global contextual information extraction network four times. Finally, the predicted probabilities of human activities are obtained by multi-layer perceptron. Numerical simulations and experiments are conducted to verify the effectiveness of the proposed method, showing that the proposed method can identify sudden abnormal human activities, and improve the recognition accuracy, generalization ability and robustness.

**Corresponding Papers:**

[1] 

## II. TWR ECHO MODEL AND PREPROCESSING METHODS

### A. Theory in Simple 

The proposed method first converts the frequency-domain echo received by the radar to the time domain first, then extracts its baseband signal, and then concatenates it along the slow time dimension, and the resulting image is a range-time map (RTM). The Doppler-time map (DTM) is obtained by summing all range bins of the RTM and doing the short time fourier transform (STFT) along the slow time dimension. The target image after clutter and noise suppression is obtained by doing Moving Target Indication Filtering (MTI) and Empirical Modal Decomposition (EMD) on both RTM and DTM, respectively. Finally, the generation of $\mathbf{R^2TM}$ and $\mathbf{D^2TM}$ is realized by vertical axis stretching.

![回波模型](https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/2073ad58-2b97-4a75-9c2b-93b5bc7b631a)
Fig. 2. Flowchart of the proposed radar data preprocessing method.

### B. Codes Explanation

**Codes Folder: ** 




