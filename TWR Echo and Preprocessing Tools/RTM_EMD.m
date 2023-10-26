%% A Program for RTM EMD Processing
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
% In this code, the Empirical Mode Decomposition (EMD) algorithm is performed 
% on the input image using the emd function. The code calculates the energy 
% of each mode and iterates through different modes to filter out low-frequency 
% components and noise. The SNR is then calculated based on the signal energy 
% and noise energy. The mode that yields the maximum SNR is chosen as the 
% optimal mode to filter the image. The resulting image will have the highest 
% SNR possible.
% 
% Citation:
% [1] P. Cao, W. Xia, M. Ye, J. Zhang, and J. Zhou, “Radar‐ID: human 
% identification based on radar micro‐Doppler signatures using deep 
% convolutional neural networks,” IET Radar, Sonar & Navigation, vol. 12, 
% no. 7, pp. 729–734, Jul. 2018.
% [2] M. B.R. and P. Rema, “A Performance based comparative study on the 
% Modified version of Empirical Mode Decomposition with traditional Empirical 
% Mode Decomposition,” Procedia Computer Science, vol. 171, pp. 2469–2475, 2020.

%% Main Program for RTM_EMD
function RTM_Processed = RTM_EMD(RTM)
    % Perform EMD.
    imf = emd(RTM);
    
    % Calculate the energy of each mode
    mode_energy = sum(imf.^2, 1);
    
    % Determine the adaptive threshold for maximum SNR.
    signal_energy = mode_energy(1);
    noise_energy = signal_energy; % Assuming noise energy equals the signal energy initially.
    max_snr = -Inf; % Initialize maximum SNR.
    
    for mode = 1:size(imf, 1)
        % Filter out low-frequency components and noise.
        filtered_image = sum(imf(1:mode, :), 1);
        
        % Calculate the noise energy as the difference between the total
        % energy and signal energy.
        noise_energy = sum(mode_energy(mode+1:end));
        
        % Calculate the SNR.
        snr = signal_energy / noise_energy;
        
        % Update the maximum SNR and corresponding mode.
        if snr > max_snr
            max_snr = snr;
            optimal_mode = mode;
        end
    end
    
    % Filter out low-frequency components and noise using the optimal mode.
    filtered_image = sum(imf(1:optimal_mode, :), 1);
    
    % Store the processed image.
    RTM_Processed = filtered_image;

end
