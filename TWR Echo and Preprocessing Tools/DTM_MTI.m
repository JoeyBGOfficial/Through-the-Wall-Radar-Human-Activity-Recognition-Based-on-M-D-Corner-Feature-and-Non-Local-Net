%% A Program for DTM Declutter Processing
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% In radar signal processing, Moving Target Indication (MTI) 
% is a commonly used algorithm to mitigate the clutter caused by stationary 
% objects in the radar environment and enhance the detection of moving targets. 
% The MTI algorithm works by exploiting the Doppler effect exhibited by moving targets.
% The MTI algorithm aims to suppress clutter returns from stationary objects 
% such as buildings, mountains, or sea clutter. 
% Since clutter returns are typically characterized by low Doppler shifts, 
% a common approach is to apply a Doppler filter to attenuate or eliminate the clutter echoes. 
% This can be achieved using a moving average filter or more sophisticated adaptive filters.
% The MTI filter used in this paper is very simple, 
% achieving suppression of stationary components by directly implementing column vector differencing in the image.
%
% Theory in Simple:
% The MTI filter applies differencing in the range-Doppler domain to suppress 
% the clutter caused by stationary targets. 
% The basic mathematical formula for the MTI filter can be represented as:
% MTI_Output(i, j) = Radar_Data(i, j) - Radar_Data(i-1, j)
% where MTI_Output represents the output of the MTI filter, 
% Radar_Data is the input radar data, and (i, j) denotes the range and Doppler bin indices, respectively.
% By subtracting the current range bin's data from the previous range bin's data, 
% the clutter caused by stationary targets is effectively attenuated since 
% their echoes remain constant over adjacent range bins.
% For the RTM and DTM mentioned in this paper, the approach for MTI is different. 
% The code provided is specifically designed for processing the DTM.
% 
% Citation:
% [1] A. Helen Victoria and G. Maragatham, "Activity recognition of FMCW 
% radar human signatures using tower convolutional neural networks,” 
% Wireless Networks, Jun. 2021.
% [2] M. Chakraborty, H. C. Kumawat, S. V. Dhavale, and A. B. Raj A, 
% “Application of DNN for radar micro-doppler signature-based human suspicious 
% activity recognition,” Pattern Recognition Letters, vol. 162, pp. 1–6, Oct. 2022.
% [3] X. Li, Y. He, and X. Jing, “A Survey of Deep Learning-Based Human 
% Activity Recognition in Radar,” Remote Sensing, vol. 11, no. 9, p. 1068, May 2019.
% [4] N. J. Mohamed, “Carrier-free radar signal design with MTI Doppler 
% processor,” IEE Proceedings - Radar, Sonar and Navigation, vol. 141, 
% no. 1, p. 59, 1994.

%% Main Program for DTM_Declutter
function DTM_Declutter = DTM_MTI(DTM)
    % Normalize pixel values.
    normalizedImage = mat2gray(DTM);

    % Get image size
    [numRows, numCols] = size(normalizedImage);

    % Initialize the decluttered image.
    DTM_Declutter = zeros(numRows, numCols-1);

    % Compute horizontal differences.
    for i = 1:numRows
        for j = 1:numCols-1
            DTM_Declutter(i, j) = normalizedImage(i, j+1) - normalizedImage(i, j);
        end
    end
    
    % Calculate the number of rows to set to zero.
    zeroRows = floor(numRows/20);
    
    % Set middle rows to zero.
    middleRowStart = floor(numRows/2) - floor(zeroRows/2);
    middleRowEnd = middleRowStart + zeroRows - 1;
    DTM_Declutter(middleRowStart:middleRowEnd, :) = 0;

end
