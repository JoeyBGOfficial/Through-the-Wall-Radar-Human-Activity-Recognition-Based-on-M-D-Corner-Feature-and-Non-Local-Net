%% A Program for Harris Corner Detector Tool
% Former Author: Team of MATLAB Computer Vision Toolbox;
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
% 
% Citation:
% [1] H. ZHANG, Y. LI, and C. CHU, “Multi-scale Harris corner detection 
% based on image block,” Journal of Computer Applications, vol. 31, no. 2, 
% pp. 356–357, Apr. 2011.

%% Harris Corner Detector Tool
function varargout = corner(varargin)
% Find corner points in an image using Harris model.

matlab.images.internal.errorIfgpuArray(varargin{:});

% parse inputs
args = matlab.images.internal.stringToChar(varargin);
[I,method,sensitivity_factor,...
    filter_coef,max_corners,quality_level] = parseInputs(args{:});

% Pass through parsed inputs to cornermetric. Only specify
% SensitivityFactor if Harris is the method. We've already verified in
% parseInputs that the user did not specify SensitivityFactor with a method
% other than Harris.
if strcmpi(method,'Harris');
    cMetric = cornermetric(I,...
        'Method',method,...
        'SensitivityFactor',sensitivity_factor,...
        'FilterCoefficients',filter_coef);
else
    cMetric = cornermetric(I,...
        'Method',method,...
        'FilterCoefficients',filter_coef);
end

% Use findpeak to do peak detection on output of cornermetric to find x/y
% locations of peaks.
[xpeak, ypeak] = findLocalPeak(cMetric,quality_level);

% Create vector of interest points from x/y peak locations.
corners = [xpeak, ypeak];

% Return N corners as specified by maxNumCorners optional input argument.
if max_corners < size(corners,1)
    corners = corners(1:max_corners,:); 
end

varargout{1} = corners;

%------------------------------------------------------------------
function BW = suppressLowCornerMetricMaxima(cMetric,BW,quality_level)

max_cmetric = max(cMetric(:));
if max_cmetric > 0
    min_metric = quality_level * max_cmetric;
else
    % Edge case: All corner metric values are 0 or less than zero. In this
    % case, mask all local maxima and return. This case arrises for uniform
    % input images.
    BW(:) = false;
    return
end

% Mask peak locations that are less than min_metric.
BW(cMetric < min_metric) = false;

%-------------------------------------------------------------------------
function [xpeak,ypeak] = findLocalPeak(cMetric,quality_level)

% The cornermetric matrix is all equal values for all input
% images with numRows/numCols <= 3. We require at least 4 rows
% and 4 columns to return a non-empty corner array.
[numRows,numCols] = size(cMetric);
if  ( (numRows < 4) || (numCols < 4))
    xpeak  = [];
    ypeak = [];
    return
end

% Find local maxima of corner metric matrix.
BW = imregionalmax(cMetric,8);

BW = suppressLowCornerMetricMaxima(cMetric,BW,quality_level);

% Suppress connected components which have same intensity and are part of
% one local maxima grouping. We want to 'thin' these local maxima to a
% single point using bwmorph.
BW = bwmorph(BW,'shrink',Inf);

% Return r/c locations that are valid non-thresholded corners. Return
% corners in order of decreasing corner metric value.
[r,c] = sortCornersByCornerMetric(BW,cMetric);

xpeak = c;
ypeak = r;

%-------------------------------------------------------------------------
function [r,c] = sortCornersByCornerMetric(BW,cMetric)

ind = find(BW);
[~,sorted_ind] = sort(cMetric(ind),'descend');
[r,c] = ind2sub(size(BW),ind(sorted_ind));

%-------------------------------------------------------------------------
function [I,method,sensitivity_factor,filter_coef,...
                  max_corners, quality_level] = parseInputs(varargin)
    
   % Reform varargin into consistent ordering: corner(I,METHOD,N) so that
   % we can use inputParser to do input parsing on optional args and P/V
   % pairs.
   if (nargin>1 && isnumeric(varargin{2}))
       if nargin>2
           %corner(I,N,P1,V1,...)
           varargin(4 : (nargin+1) ) = varargin(3:nargin);
       end
       
       %Now that any P/V have been handled, reorder 2nd and 3rd arguments
       %into the form:
       %corner(I,METHOD,N,...)
      varargin{3} = varargin{2};
      varargin{2} = 'Harris';
      
   end
       
parser = commonCornerInputParser(mfilename);
parser.addOptional('N',200,@checkMaxCorners);
parser.addParamValue('QualityLevel',0.01,@checkQualityLevel);

% parse input
parser.parse(varargin{:});

% assign outputs
I = parser.Results.Image;
method = parser.Results.Method;
sensitivity_factor = parser.Results.SensitivityFactor;
filter_coef = parser.Results.FilterCoefficients;
max_corners = parser.Results.N;
quality_level = parser.Results.QualityLevel;

% check for incompatible parameters.  if user has specified a sensitivity
% factor with method other than harris, we error.  We made the sensitivity
% factor default value a string to determine if one was specified or if the
% default was provided since we cannot get this information from the input
% parser object.
method_is_not_harris = ~strcmpi(method,'Harris');
sensitivity_factor_specified = ~ischar(sensitivity_factor);
if method_is_not_harris && sensitivity_factor_specified
    error(message('images:corner:invalidParameterCombination'));
end

% convert from default strings to actual values.
if ischar(sensitivity_factor)
    sensitivity_factor = str2double(sensitivity_factor);
end

%-------------------------------
function tf = checkMaxCorners(x)
        
validateattributes(x,{'numeric'},{'nonempty','nonnan','real',...
            'scalar','integer','positive','nonzero'},mfilename,'N');
tf = true;

%-------------------------------
function tf = checkQualityLevel(x)
        
validateattributes(x,{'numeric'},{'nonempty','nonnan','real',...
            'scalar'},mfilename,'QualityLevel');
        
% The valid range is (0,1).        
if (x >=1) || (x <=0)
    error(message('images:corner:invalidQualityLevel'));
end

tf = true;
