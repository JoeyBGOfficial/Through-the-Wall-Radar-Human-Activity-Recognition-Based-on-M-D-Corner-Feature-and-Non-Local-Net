%% A Program for Channel-Direction Concatenation Processing
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-16;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% In this MATLAB function Channel_Concatenation, the two input images 
% R2TM_Corners and D2TM_Corners are expected to be three-channel color images.
% The function first resizes the input images to a size of 256 x 256 pixels 
% using the imresize function.
% Next, the resized images R2TM_Corners_resized and D2TM_Corners_resized 
% are concatenated along the channel dimension using the cat function with 
% the argument 3 indicating the channel dimension.
% The resulting concatenated image is stored in the variable 
% concatenated_image, which is then returned as the output of the function.
% You can use this Channel_Concatenation function in MATLAB by passing the 
% two input images as arguments, and it will return the concatenated image 
% with the resized dimensions of 256x256 pixels.
%
% Citation: None.

%% Main Body for Channel_Concatenation
function concatenated_image = Channel_Concatenation(R2TM_Corners, D2TM_Corners)
    % Concatenate two three-channel color images along the channel
    % dimension.
    
    % Resize the images to 256x256.
    R2TM_Corners_resized = imresize(R2TM_Corners, [256, 256]);
    D2TM_Corners_resized = imresize(D2TM_Corners, [256, 256]);
    
    % Concatenate the resized images along the channel dimension.
    concatenated_image = cat(3, R2TM_Corners_resized, D2TM_Corners_resized);

end
