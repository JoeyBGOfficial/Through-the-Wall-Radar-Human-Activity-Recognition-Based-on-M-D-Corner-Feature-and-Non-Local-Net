%% A Program for R2TM Corner Detection Using Harris Model
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-17;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The Harris corner detection algorithm is a widely used method for identifying 
% corners in an image. It was proposed by Chris Harris and Mike Stephens in 1988. 
% The algorithm is based on the observation that corners have significant 
% intensity variations in all directions.
%
% Theory in Simple:
% The context below is a detailed explanation of the Harris corner detection 
% algorithm:
% 1. Convert the Image to Grayscale:
% The algorithm begins by converting the input image to grayscale. 
% This is done to simplify the corner detection process, as corners can be 
% defined by changes in intensity rather than color.
% 2. Compute Image Gradients:
% The next step involves computing the gradients of the grayscale image. 
% This is achieved by applying a gradient operator, such as the Sobel operator, 
% to estimate the intensity changes in both the horizontal and vertical directions. 
% The gradients provide information about the intensity variations in different directions.
% 3. Compute the Harris Response Function:
% The Harris response function is the core of the algorithm. It measures 
% the likelihood of a pixel being a corner based on the gradients of the image. 
% The response function at each pixel is calculated using the following formula:
% R = det(M) - k * trace(M)^2
% where:
% R is the response value at a given pixel.
% M is the structure tensor, which is a matrix computed from the gradients.
% det(M) and trace(M) represent the determinant and trace of the structure 
% tensor, respectively.
% k is an empirical constant typically set in the range of 0.04 to 0.06. 
% It helps adjust the sensitivity of the detector.
% The response value R is positive for corners, negative for edges, 
% and close to zero for flat regions.
% 4. Thresholding and Non-maximum Suppression:
% To extract reliable corners, the response values are thresholded to select 
% pixels with sufficiently high values. A common approach is to choose a 
% threshold value manually or using an automatic method like Otsu's thresholding.
% After thresholding, non-maximum suppression is applied. This step ensures 
% that only local maxima in the response function are considered as corners. 
% It involves comparing each pixel's response value with its neighborhood 
% and suppressing non-maximum values.
% 5. Marking the Detected Corners:
% Once the corners are identified, they can be marked on the original image 
% for visualization purposes. This is often done using markers or circles to 
% indicate the detected corner locations.
% The following is the code for the R2TM_Corner_Detector function that takes 
% a grayscale image R2TM as input, detects 30 corners using the Harris corner 
% detector, and outputs an image R2TM_Corners with only the detected corners:
% The corner function is used to apply the Harris corner detection algorithm 
% to the input grayscale image. The second argument of the corner function 
% is set to 30, indicating that we want to detect 30 corners in total.
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
function R2TM_Corners = R2TM_Corner_Detector(R2TM)
    % Detect 30 corners on R2TM using the Harris corner detector.
    
    % Apply the Harris corner detection algorithm.
    corners = corner(R2TM, 30);
    
    % Create an image with only the detected corners.
    R2TM_Corners = zeros(size(R2TM));
    num_corners = size(corners, 1);
    for i = 1:num_corners
        x = corners(i, 2);
        y = corners(i, 1);
        R2TM_Corners(round(x), round(y)) = 255;
    end
    
    % Convert the image to uint8 format.
    R2TM_Corners = uint8(R2TM_Corners);

end
