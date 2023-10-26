%% A Program for DTM Generator
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The Short-Time Fourier Transform (STFT) is a time-frequency analysis 
% technique used to analyze and visualize the local frequency content of a 
% signal over time. It provides a way to examine how the frequency components 
% of a signal change as time progresses.
%
% Theory in Simple:
% Here's a detailed explanation of the radar Doppler time image generation 
% method based on the STFT:
% 1. Data Acquisition: 
% Firstly, the raw radar data is obtained through a radar system. This data 
% typically consists of a series of echo signals that include the reflections 
% from target objects.
% 2. Preprocessing: 
% The raw radar data undergoes necessary preprocessing steps, such as noise 
% removal, filtering, and signal enhancement. This step helps improve the 
% effectiveness of subsequent processing.
% 3. STFT Computation: 
% The preprocessed radar data is divided into multiple windows, and the STFT 
% algorithm is applied to each window. STFT is a time-frequency analysis 
% method that localizes the signal in both the time and frequency domains. 
% It segments the signal into short time frames using a window function and 
% then applies the Fourier transform to each frame, yielding the frequency 
% spectrum for that frame.
% 4. Spectrum Processing: 
% Further processing is applied to the frequency spectra obtained from each 
% window. Common processing techniques include filtering, smoothing, and 
% enhancement to extract relevant frequency features or remove unwanted noise.
% 5. Doppler Frequency Shift Extraction: 
% The Doppler frequency shift information is extracted from the processed 
% frequency spectra. Doppler frequency shift occurs due to the motion of 
% target objects relative to the radar system. By analyzing the frequency 
% component variations in the spectra, the target's velocity and direction 
% of motion can be determined.
% 6 Time Image Generation: 
% Based on the Doppler frequency shift information, it is mapped onto the 
% time axis to generate the radar Doppler time image. In the time image, 
% the horizontal axis represents time, and the vertical axis represents the 
% range or distance. The intensity or color at each point in the image 
% corresponds to the Doppler frequency shift or target velocity at that 
% particular time and range.
% In this code, the input image RTM is summed along the columns to obtain a 
% row vector named summed_vector. The window size and overlap size for the 
% STFT are determined to achieve maximum time-frequency resolution. The 
% desired maximum resolution is specified as max_resolution, which you can 
% adjust according to your requirements. The window size is calculated based 
% on the desired maximum resolution, and the overlap size is set to half of 
% the window size. The code applies the STFT using the determined window and 
% overlap sizes and obtains the magnitude spectrogram Z. Finally, the absolute 
% value of Z is taken to obtain the DTM image, which is stored in the variable DTM.
% 
% Citation:
% [1] P. van Dorp and F. C. A. Groen, “Human walking estimation with radar,” 
% IEE Proceedings - Radar, Sonar and Navigation, vol. 150, no. 5, p. 356, 2003.
% [2] S. G. Bhatti and A. I. Bhatti, “Radar Signals Intrapulse Modulation 
% Recognition Using Phase-Based STFT and BiLSTM,” IEEE Access, vol. 10, 
% pp. 80184–80194, 2022.
% [3] X. Mou, X. Chen, N. Su, and J. Guan, “Motion classification for radar 
% moving target via STFT and convolution neural network,” The Journal of 
% Engineering, vol. 2019, no. 19, pp. 6287–6290, Jul. 2019.

%% Main Body of DTM Generator
function DTM = DTM_Generator(RTM, max_resolution)
    % Sum the columns of the input image to obtain a row vector.
    summed_vector = sum(RTM, 2)';
    
    % Determine the window and overlap sizes for maximum time-frequency
    % resolution.
    vector_length = length(summed_vector);
    % max_resolution = 2; % Maximum desired time-frequency resolution, adjust as needed.
    
    % Calculate the window size based on the desired maximum resolution.
    window_size = round(vector_length / max_resolution);
    window_size = 2^nextpow2(window_size); % Ensure window size is a power of 2.
    
    % Calculate the overlap size to achieve maximum time-frequency
    % resolution.
    overlap_size = window_size / 2;
    
    % Apply the STFT using the determined window and overlap sizes.
    [~, ~, Z] = spectrogram(summed_vector, window_size, overlap_size);
    
    % Take the absolute value of the STFT result to obtain the DTM image.
    DTM = abs(Z);

end
