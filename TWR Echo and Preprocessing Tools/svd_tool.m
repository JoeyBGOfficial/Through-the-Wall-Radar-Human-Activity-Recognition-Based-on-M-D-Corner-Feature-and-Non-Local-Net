%% A Program for SVD Tool
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% SVD is a widely used technique in radar signal processing for various 
% applications such as target detection, parameter estimation, and adaptive 
% beamforming. It provides a way to decompose a matrix into three separate 
% matrices, enabling the analysis and manipulation of the data contained 
% within the matrix.
%
% Theory in Simple:
% Let's consider a matrix X of size M x N, where M represents the number of 
% samples (range bins) and N represents the number of pulses (or time snapshots). 
% Each element x[m, n] of the matrix represents the received signal at range 
% bin m and time snapshot n.
% The SVD of matrix X can be expressed as:
% X = U * Σ * V^H
% where:
% U is an M x M unitary matrix, representing the left singular vectors of X.
% Σ is an M x N rectangular diagonal matrix, with non-negative real numbers 
% on its diagonal, known as the singular values of X.
% V^H is the conjugate transpose of an N x N unitary matrix V, representing 
% the right singular vectors of X.
% The singular values in the diagonal matrix Σ are arranged in descending order, 
% indicating their significance in describing the data. The left singular 
% vectors U and right singular vectors V correspond to the singular values 
% and provide information about the spatial and temporal characteristics of 
% the radar data, respectively.
% 
% Citation:
% [1] S. Wu, Q. Huang, J. Chen, S. Meng, G. Fang, and H. Yin, “Target 
% Localization and Identification Algorithm for Ultra Wideband Through-wall 
% Radar,” Journal of Electronics & Information Technology, vol. 32, no. 11, 
% pp. 2624–2629, Dec. 2010.
% [2] F. Fioranelli, M. Ritchie, and H. Griffiths, “Performance Analysis of 
% Centroid and SVD Features for Personnel Recognition Using Multistatic 
% Micro-Doppler,” IEEE Geoscience and Remote Sensing Letters, vol. 13, no. 5,
% pp. 725–729, May 2016.

%% SVD Tool
function [U, S, V] = svd_tool(A)
    % Perform SVD of matrix A.
    
    % Compute the eigenvalues and eigenvectors of A*A'.
    [V, D] = eig(A * A');
    
    % Sort the eigenvalues in descending order.
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Compute the singular values (square roots of eigenvalues).
    singular_values = sqrt(diag(D(idx, idx)));
    
    % Compute the singular value matrix S.
    S = diag(singular_values);
    
    % Compute the matrix U.
    U = A' * V / S;

end
