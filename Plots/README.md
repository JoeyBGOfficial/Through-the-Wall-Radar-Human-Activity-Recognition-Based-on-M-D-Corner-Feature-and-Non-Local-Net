## II. TWR ECHO MODEL AND PREPROCESSING METHODS

### A. Theory in Simple 

The proposed method first converts the frequency-domain echo received by the radar to the time domain first, then extracts its baseband signal, and then concatenates it along the slow time dimension, and the resulting image is a range-time map (RTM). The Doppler-time map (DTM) is obtained by summing all range bins of the RTM and doing the short time fourier transform (STFT) along the slow time dimension. The target image after clutter and noise suppression is obtained by doing Moving Target Indication Filtering (MTI) and Empirical Modal Decomposition (EMD) on both RTM and DTM, respectively. Finally, the generation of $\mathbf{R^2TM}$ and $\mathbf{D^2TM}$ is realized by vertical axis stretching.

![回波模型](https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/2073ad58-2b97-4a75-9c2b-93b5bc7b631a)
Fig. 2. Flowchart of the proposed radar data preprocessing method.

### B. Codes Explanation (Folder: TWR Echo and Preprocessing Tools, Plots)

#### 1. Channel_Concatenation.m:

The function is used to concatenate color mapped $\mathbf{R^2TM}$ and $\mathbf{D^2TM}$ in channel direction.

**Input:** $\mathbf{R^2TM}$ and $\mathbf{D^2TM}$, with the size of $256 \times 256 \times 3$, respectively.

**Output:** Concatenated image with the size of $256\times 256\times 6$.

#### 2. Convert_Gray_to_RGB.m:

This function implements pseudo-color mapping for any input of grayscale image.

**Input:** Gray-scale image $\mathbf{I}$.

**Output:** Color image $\mathbf{I}_\mathrm{RGB}$.

#### 3. DTM_EMD.m:

This function implements the empirical modal decomposition processing on the DTM.

**Input:** $\mathbf{DTM}$ in matrix form.

**Output:** $\mathbf{DTM}_\mathrm{Processed}$.

#### 4. DTM_Generator.m:

This function implements the time-frequency analysis using STFT.

**Input:** $\mathbf{RTM}$ in matrix form, and max_resolution, which adjusts the resolution of frequency domain in the result of STFT.

**Output:** $\mathbf{DTM}$.

#### 5. DTM_MTI.m:

This function implements the moving target indication filter of DTM.

**Input:** $\mathbf{DTM}$ in matrix form.

**Output:** $\mathbf{DTM}_\mathrm{Declutter}$.

#### 6. DTM_SVD.m:

This function implements the sigular value decomposition filter of DTM.

**Input:** $\mathbf{DTM}$ in matrix form.

**Output:** $\mathbf{DTM}_\mathrm{Processed}$.

#### 7. RTM_EMD.m:

This function implements the empirical modal decomposition processing on the RTM.

**Input:** $\mathbf{RTM}$ in matrix form.

**Output:** $\mathbf{RTM}_\mathrm{Processed}$.

#### 8. RTM_MTI.m:

This function implements the moving target indication filter of RTM.

**Input:** $\mathbf{RTM}$ in matrix form.

**Output:** $\mathbf{RTM}_\mathrm{Declutter}$.

#### 9. RTM_SVD.m:

This function implements the sigular value decomposition filter of RTM.

**Input:** $\mathbf{RTM}$ in matrix form.

**Output:** $\mathbf{RTM}_\mathrm{Processed}$.

#### 10. emd.m:

This function is a tool function for the empirical modal decomposition algorithm. Sourced from MATLAB's native toolbox and fine-tuned.

**Input:** $\mathbf{x}$ in sequence signal form.

**Output:** Decomposed IMFs and variables of EMD processing corresponding to input $\mathbf{x}$.

#### 11. ind2rgb_tool.m:

This function is a tool function for the pseudo-color mapping algorithm. Sourced from MATLAB's native toolbox and fine-tuned.

**Input:** $\mathbf{Indexed}$ in 2D integer matrix form, and $\mathrm{Colormap}$ object in MATLAB standard colormap form, which should be an M-By-3 matrix.

**Output:** Mapped image $\mathbf{RGB}$.

#### 12. nextpow2.m:

This function is a implementation of finding the next neighboring power of $2$ of a input number. Sourced from MATLAB's native toolbox and fine-tuned.

**Input:** Number $n$.

**Output:** Neighboring power $p$ of $2$.

#### 13. svd_tool.m:

Main code of the singular value decomposition algorithm. Sourced from MATLAB's native toolbox and fine-tuned.

**Input:** Matrix $\mathbf{A}$.

**Output:** The left chord vectors $\mathbf{U}$, the singular value matrix $\mathbf{S}$, and the right chord vectors $\mathbf{V}$.

#### 14. D2TM_Generator.m:

Code for generating $\mathbf{D^2TM}$ by vertical axis stretching.

**Input:** Matrix $\mathbf{DTM}$.

**Output:** Stretched matrix $\mathbf{D^2TM}$.

#### 15. R2TM_Generator.m:

Code for generating $\mathbf{R^2TM}$ by vertical axis stretching.

**Input:** Matrix $\mathbf{RTM}$.

**Output:** Stretched matrix $\mathbf{R^2TM}$.

### C. Datafiles Explanation (Folder: TWR Echo and Preprocessing Tools, Plots)

#### 1. R2TM_D2TM_Clist.mat:

This data file stores the color maps used to generate $\mathbf{R^2TM}$ and $\mathbf{D^2TM}$, which is a tool that corresponds to the color maps in the paper.
