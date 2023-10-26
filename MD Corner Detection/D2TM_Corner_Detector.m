%% A Program for D2TM Corner Detection Using Moravec Model
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-17;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The Moravec corner detection algorithm is a popular method for identifying 
% corners in an image. It was proposed by Chris Moravec in 1980. The algorithm 
% is based on the observation that corners exhibit large intensity variations 
% in different directions.
%
% Theory in Simple:
% The context below is a detailed explanation of the Moravec corner detection 
% algorithm:
% 1. Convert the Image to Grayscale:
% The algorithm begins by converting the input image to grayscale. 
% This is done to simplify the corner detection process, as corners can be 
% defined by changes in intensity rather than color.
% 2. Compute Local Intensity Variations:
% The next step involves calculating the intensity variations within small 
% window regions of the grayscale image. The algorithm compares the intensity 
% values of each pixel with its neighboring pixels in different directions 
% (up, down, left, and right) by shifting the window. The differences between 
% the intensities are squared and summed to measure the local intensity variation.
% 3. Compute the Moravec Response Function:
% The Moravec response function is the core of the algorithm. It measures 
% the likelihood of a pixel being a corner based on the local intensity variations. 
% The response value at each pixel is calculated by finding the minimum intensity 
% variation among the different window shifts. This is done using the following formula:
% R = min(min(Euclidean_distance))
% where:
% R is the response value at a given pixel.
% Euclidean_distance represents the squared intensity differences between 
% the central pixel and its shifted neighbors.
% The response value R is low for corner pixels with significant intensity 
% variations and high for non-corner pixels with relatively small intensity 
% differences.
% 4. Thresholding and Non-maximum Suppression:
% To extract reliable corners, the response values are thresholded to select 
% pixels with sufficiently high values. A common approach is to choose a 
% threshold value manually or using an automatic method like Otsu's thresholding.
% After thresholding, non-maximum suppression is applied. This step ensures 
% that only local maxima in the response function are considered as corners. 
% It involves comparing each pixel's response value with its neighboring 
% pixels and suppressing non-maximum values.
% 5. Marking the Detected Corners:
% Once the corners are identified, they can be marked on the original image 
% for visualization purposes. This is often done using markers or circles 
% to indicate the detected corner locations.
% The following is the MATLAB function code for the D2TM_Corner_Detector function 
% that takes a grayscale image D2TM as input, detects 22 corners using the 
% Moravec corner detector, and outputs an image D2TM_Corners with only the 
% detected corners.
%
% Citation:
% [1] H. ZHANG, Y. LI, and C. CHU, “Multi-scale Harris corner detection 
% based on image block,” Journal of Computer Applications, vol. 31, no. 2, 
% pp. 356–357, Apr. 2011.
% [2] M. Trajković and M. Hedley, "Fast corner detection,” Image and Vision 
% Computing, vol. 16, no. 2, pp. 75–87, Feb. 1998.
% [3] J. Jing, C. Liu, W. Zhang, Y. Gao, and C. Sun, “ECFRNet: Effective 
% corner feature representations network for image corner detection,” 
% Expert Systems with Applications, vol. 211, p. 118673, Jan. 2023.

%% Main Body of R2TM_Corner_Detector
function D2TM_Corners = D2TM_Corner_Detector(D2TM)
    % Detect 22 corners on D2TM using the Moravec corner detector.
    
    % Set the window size for local intensity comparison.
    window_size = 3;
    
    % Compute the image gradients.
    [Gx, Gy] = gradient(double(D2TM));
    
    % Calculate the Moravec response function.
    response = zeros(size(D2TM));
    for i = 2:size(D2TM, 1)-1
        for j = 2:size(D2TM, 2)-1
            % Calculate the sum of squared differences (SSD) for each
            % window shift.
            SSD = zeros(4, 1);
            SSD(1) = sum(sum((D2TM(i-1:i+1, j-1:j+1) - D2TM(i-1:i+1, j)).^2));
            SSD(2) = sum(sum((D2TM(i-1:i+1, j-1:j+1) - D2TM(i-1:i+1, j+1)).^2));
            SSD(3) = sum(sum((D2TM(i-1:i+1, j-1:j+1) - D2TM(i, j-1:j+1)).^2));
            SSD(4) = sum(sum((D2TM(i-1:i+1, j-1:j+1) - D2TM(i+1, j-1:j+1)).^2));
            
            % Calculate the minimum SSD value as the response.
            response(i, j) = min(SSD);
        end
    end
    
    % Select the top 22 corners based on the response values.
    [~, indices] = sort(response(:), 'descend');
    corner_indices = indices(1:22);
    [corner_rows, corner_cols] = ind2sub(size(D2TM), corner_indices);
    
    % Create an image with only the detected corners.
    D2TM_Corners = zeros(size(D2TM));
    for k = 1:length(corner_indices)
        D2TM_Corners(corner_rows(k), corner_cols(k)) = 255;
    end
    
    % Convert the image to uint8 format.
    D2TM_Corners = uint8(D2TM_Corners);

end
