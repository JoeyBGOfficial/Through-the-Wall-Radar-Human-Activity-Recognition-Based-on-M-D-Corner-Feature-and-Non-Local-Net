%% A Program for ind2rgb Tool
% Former Author: Team of MATLAB Image Processing Toolbox;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-17;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The ind2rgb function in MATLAB is a built-in function that converts an 
% indexed image to an RGB image using a specified colormap. Here we show 
% you a simplified MATLAB implementation that achieves a similar
% functionality.
%
% Citation: None.

%% ind2rgb Tool
function RGB = ind2rgb_tool(Indexed, Colormap)
    % Convert an indexed image to an RGB image using a specified colormap.
    
    % Validate the input colormap.
    if size(Colormap, 2) ~= 3
        error('Colormap must be an M-by-3 matrix.');
    end

    % Validate the indexed image.
    if ~isinteger(Indexed) || ~ismatrix(Indexed)
        error('Indexed must be a 2D matrix of integer values.');
    end
    
    % Check the range of the indexed image.
    if any(Indexed(:) < 1) || any(Indexed(:) > size(Colormap, 1))
        error('Indexed values are out of range for the specified colormap.');
    end

    % Initialize the RGB image.
    RGB = zeros([size(Indexed), 3], 'like', Indexed);

    % Map the indexed image to RGB values using the colormap.
    for channel = 1:3
        RGB(:,:,channel) = Colormap(Indexed, channel);
    end
   
end
