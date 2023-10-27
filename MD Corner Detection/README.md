## III. MICRO-DOPPLER CORNER DETECTION METHODS

### A. Theory in Simple 

The proposed method utilizes Harris model based detector to extract corner features on $\mathbf{R^2TM}$ and Moravec model based detector to extract corner features on $\mathbf{D^2TM}.

![微信截图_20231022155040](https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/381900b4-9590-497e-a9ce-2bb0c373190b)

Fig. 3. Detecting corner points on two types of radar images using two different detectors, respectively.

### B. Codes Explanation (Folder: MD Corner Detection)

#### 1. D2TM_Corner_Detector.m:

The function is used to achieve Moravec model based corner detection on $\mathbf{D^2TM}$.

**Input:** $\mathbf{D^2TM}$ in matrix form.

**Output:** Corner feature map of $\mathbf{D^2TM}$.

#### 2. R2TM_Corner_Detector.m:

The function is used to achieve Moravec model based corner detection on $\mathbf{R^2TM}$.

**Input:** $\mathbf{R^2TM}$ in matrix form.

**Output:** Corner feature map of $\mathbf{R^2TM}$.

#### 3. corner.m:

The function is the main body of the Harris model based corner detection algorithm. Sourced from MATLAB's native toolbox and fine-tuned.

**Input:** Image for detection.

**Output:** Harris based corner feature map and variables corresponding to the input image.

#### 4. gradient.m:

The function is the tool for calculating gradient on a vector or an image.

**Input:** Signal or image.

**Output:** Gradient map corresponding to the input signal or image.

### C. Datafiles Explanation (Folder: MD Corner Detection)

None.
