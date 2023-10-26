%% A Program for Gray-Scale Map to RGB Converting
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-17;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% In this MATLAB function Convert_Corners_to_RGB, the input image I is 
% expected to be a single-channel grayscale image.
% The function starts by applying a pseudocolor map to the grayscale image 
% using the ind2rgb function. The jet(256) colormap is used, which generates 
% a colormap with 256 colors ranging from blue to red. Each grayscale intensity 
% value in the input image is mapped to a corresponding color value based on 
% its position in the colormap.
% Next, the resulting RGB image I_RGB is in the range of [0, 1]. To convert 
% it to the range of [0, 255], the image is multiplied by 255 and then 
% converted to the uint8 data type using the uint8 function.
% The output of the function is the converted RGB image I_RGB, which is 
% returned as the output.
% You can use this Convert_Corners_to_RGB function in MATLAB by passing the 
% grayscale image I as an argument, and it will return the corresponding 
% RGB image where the grayscale values are represented using pseudocolors.
%
% Citation: None.

%% Main Body of Convert_Gray_to_RGB
function I_RGB = Convert_Gray_to_RGB(I)
    % Convert the grayscale image I to RGB using a pseudocolor map.
    
    % Apply a pseudocolor map to the grayscale image.
    I_RGB = ind2rgb(I, jet(256));
    
    % Convert the image to the range of [0, 255].
    I_RGB = uint8(I_RGB * 255);

end