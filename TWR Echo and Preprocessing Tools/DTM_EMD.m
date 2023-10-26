%% A Program for DTM EMD Processing
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The Empirical Mode Decomposition (EMD) algorithm is a data-driven signal 
% processing technique that decomposes a given signal into a set of intrinsic 
% mode functions (IMFs). It was developed by Huang et al. in the late 1990s 
% as a method for analyzing nonlinear and non-stationary signals.
% The key idea behind EMD is to decompose a signal into components that have 
% well-defined instantaneous frequencies. These components, known as IMFs, 
% are derived directly from the signal without requiring any predefined basis 
% functions or assumptions about the signal's properties.
%
% Theory in Simple:
% The EMD algorithm follows these steps:
% 1. Start with the given signal as the first IMF candidate, called the 
% "original signal."
% 2. Identify all the local extrema (maxima and minima) in the original signal.
% 3. Interpolate the upper envelope by connecting the maxima using cubic 
% spline interpolation. Similarly, interpolate the lower envelope by 
% connecting the minima.
% 4. Compute the mean value of the upper and lower envelopes to obtain the 
% "mean envelope."
% 5. Subtract the mean envelope from the original signal to obtain the first IMF.
% 6. If the obtained IMF satisfies two convergence criteria (the number of 
% extrema equals or differs by at most one and the mean envelope is effectively 
% zero), it is considered a valid IMF. Otherwise, it is treated as a new signal, 
% and the process is repeated iteratively until a valid IMF is obtained.
% 7. Repeat steps 2-6 on the residual signal (original signal minus the 
% obtained IMF) to extract the next IMF.
% 8. Continue this process until the residual becomes a monotonic function 
% or exhibits a low-pass behavior, indicating the extraction of all relevant IMFs.
% 9. The final residual is considered the trend or the low-frequency component 
% of the original signal.
% In this code, the input image DTM is split along the row direction into 
% two halves, top_half and bottom_half. The Empirical Mode Decomposition 
% (EMD) algorithm is then applied to each half separately, resulting in 
% top_imf and bottom_imf, which represent the intrinsic mode functions (IMFs) 
% for each half. The code calculates the energy of each mode for both halves 
% and iteratively determines the optimal mode that maximizes the signal-to-noise 
% ratio (SNR). The low-frequency components and noise are filtered out using 
% the optimal mode for each half. Finally, the filtered halves are concatenated 
% vertically to reconstruct the processed image, stored in DTM_Processed.
% 
% Citation:
% [1] P. Cao, W. Xia, M. Ye, J. Zhang, and J. Zhou, “Radar‐ID: human 
% identification based on radar micro‐Doppler signatures using deep 
% convolutional neural networks,” IET Radar, Sonar & Navigation, vol. 12, 
% no. 7, pp. 729–734, Jul. 2018.
% [2] M. B.R. and P. Rema, “A Performance based comparative study on the 
% Modified version of Empirical Mode Decomposition with traditional Empirical 
% Mode Decomposition,” Procedia Computer Science, vol. 171, pp. 2469–2475, 2020.

%% Main Program for DTM_EMD
function DTM_Processed = DTM_EMD(DTM)
    % Split the image along the row direction into two halves.
    [rows, cols] = size(DTM);
    half_rows = floor(rows/2);
    top_half = DTM(1:half_rows, :);
    bottom_half = DTM(half_rows+1:end, :);
    
    % Perform EMD on the top half.
    top_imf = emd(top_half);
    
    % Perform EMD on the bottom half.
    bottom_imf = emd(bottom_half);
    
    % Calculate the energy of each mode for the top half.
    top_mode_energy = sum(top_imf.^2, 2);
    
    % Calculate the energy of each mode for the bottom half.
    bottom_mode_energy = sum(bottom_imf.^2, 2);
    
    % Determine the adaptive threshold for maximum SNR for the top half.
    top_signal_energy = top_mode_energy(1);
    top_noise_energy = top_signal_energy; % Assuming noise energy equals the signal energy initially.
    top_max_snr = -Inf; % Initialize maximum SNR.
    
    for mode = 1:size(top_imf, 1)
        % Filter out low-frequency components and noise for the top half.
        top_filtered_image = sum(top_imf(1:mode, :), 1);
        
        % Calculate the noise energy as the difference between the total
        % energy and signal energy.
        top_noise_energy = sum(top_mode_energy(mode+1:end));
        
        % Calculate the SNR for the top half.
        top_snr = top_signal_energy / top_noise_energy;
        
        % Update the maximum SNR and corresponding mode for the top half.
        if top_snr > top_max_snr
            top_max_snr = top_snr;
            top_optimal_mode = mode;
        end
    end
    
    % Determine the adaptive threshold for maximum SNR for the bottom half.
    bottom_signal_energy = bottom_mode_energy(1);
    bottom_noise_energy = bottom_signal_energy; % Assuming noise energy equals the signal energy initially.
    bottom_max_snr = -Inf; % Initialize maximum SNR.
    
    for mode = 1:size(bottom_imf, 1)
        % Filter out low-frequency components and noise for the bottom
        % half.
        bottom_filtered_image = sum(bottom_imf(1:mode, :), 1);
        
        % Calculate the noise energy as the difference between the total
        % energy and signal energy.
        bottom_noise_energy = sum(bottom_mode_energy(mode+1:end));
        
        % Calculate the SNR for the bottom half.
        bottom_snr = bottom_signal_energy / bottom_noise_energy;
        
        % Update the maximum SNR and corresponding mode for the bottom
        % half.
        if bottom_snr > bottom_max_snr
            bottom_max_snr = bottom_snr;
            bottom_optimal_mode = mode;
        end
    end
    
    % Filter out low-frequency components and noise using the optimal mode
    % for the top half.
    top_filtered_image = sum(top_imf(1:top_optimal_mode, :), 1);
    
    % Filter out low-frequency components and noise using the optimal mode
    % for the bottom half.
    bottom_filtered_image = sum(bottom_imf(1:bottom_optimal_mode, :), 1);
    
    % Concatenate the filtered halves to reconstruct the processed image.
    DTM_Processed = [top_filtered_image; bottom_filtered_image];

end
