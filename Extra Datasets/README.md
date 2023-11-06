## VII. EXTRA DATASETS

### A. Basic Information

 $\textbf{\color{red}{First, we need to emphasize that we do not own any rights to these datasets except for experiments!}}$

The datasets here include DIAT-μ RadHAR, an open-source dataset for radar-based human activity recognition in free space, and CI4R, an open-source dataset for small sample radar-based human activity recognition in free space. DIAT-μ RadHAR contains a total of $3780$ datas for $6$ categories of typical human activities and CI4R contains a total of $735$ datas for $11$ categories of typical human activities. In our paper, we only extracted a portion of the dataset for training and validation, with the aim of keeping the amount of data consistent across different categories.

If you want to use the DIAT-μ RadHAR dataset, please cite the original authors and paper:

[1] M. Chakraborty, H. C. Kumawat, S. V. Dhavale and A. A. B. Raj, "DIAT-μ RadHAR (Micro-Doppler Signature Dataset) & μ RadNet (A Lightweight DCNN)—For Human Suspicious Activity Recognition," in IEEE Sensors Journal, vol. 22, no. 7, pp. 6851-6858, 1 April1, 2022.

If you want to use the CI4R dataset, please cite the original authors and paper and star their Github repository:

[2] Sevgi Z. Gurbuz, M. Mahbubur Rahman, Emre Kurtoglu, Trevor Macks, and Francesco Fioranelli "Cross-frequency training with adversarial learning for radar micro-Doppler signature classification (Rising Researcher)", Proc. SPIE 11408, Radar Sensor Technology XXIV, 114080A (11 May 2020). https://github.com/ci4r/CI4R-Activity-Recognition-datasets.

### B. Codes Explanation (Folder: Extra Datasets)

#### 1. Plot_SimHSet_CI4R.m:

This script is used to read in and display images from the CI4R dataset.

#### 2. Plot_SimHSet_DIAT_JPG.m:

This script is used to read in and display images from the DIAT-μ RadHAR dataset in image format.

#### 3. Plot_SimHSet_DIAT_MAT.m:

This script is used to read in and display images from the DIAT-μ RadHAR dataset in .mat file format.

#### 4. Main.m:

The script selects the dataset you want to call via a unified dialog box and opens the corresponding above program to read, write and display the data.
