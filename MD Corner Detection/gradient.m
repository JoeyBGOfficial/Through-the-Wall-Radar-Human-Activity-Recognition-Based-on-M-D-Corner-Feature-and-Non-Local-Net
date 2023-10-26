%% A Program for Gradient Computing Tool
% Former Author: Team of MATLAB Computer Vision Toolbox;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-17;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The gradient function in MATLAB is a built-in function that computes the 
% gradient of an input matrix or vector. The gradient represents the rate 
% of change of the values in the input with respect to their spatial or 
% sequential positions.
% To use this function, try:
% [Gx, Gy] = gradient(F)
% where `F` is the input matrix or vector, and `Gx` and `Gy` are the output 
% matrices representing the gradients in the x and y directions, respectively.
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

%% Image Gradient Computing Tool
function varargout = gradient(f,varargin)
% Approximate gradient calculator.

[f,ndim,loc,rflag] = parse_inputs(f,varargin);
nargoutchk(0,ndim);

% Loop over each dimension. 
varargout = cell(1,ndim);
siz = size(f);

% first dimension 
prototype = real(full(f([])));
g = zeros(siz, 'like', prototype); % case of singleton dimension
h = loc{1}(:);
n = siz(1);
% Take forward differences on left and right edges
if n > 1
   g(1,:) = (f(2,:) - f(1,:))/(h(2)-h(1));
   g(n,:) = (f(n,:) - f(n-1,:))/(h(end)-h(end-1));
end

% Take centered differences on interior points
if n > 2
   g(2:n-1,:) = (f(3:n,:)-f(1:n-2,:)) ./ (h(3:n) - h(1:n-2));
end

varargout{1} = g;

% second dimensions and beyond
if ndim == 2
    % special case 2-D matrices to support sparse matrices,
    % which lack support for N-D operations including reshape
    % and indexing
    n = siz(2);
    h = reshape(loc{2},1,[]);
    g = zeros(siz, 'like', prototype);
    
    % Take forward differences on left and right edges
    if n > 1
        g(:,1) = (f(:,2) - f(:,1))/(h(2)-h(1));
        g(:,n) = (f(:,n) - f(:,n-1))/(h(end)-h(end-1));
    end
    
    % Take centered differences on interior points
    if n > 2
        h = h(3:n) - h(1:n-2);
        g(:,2:n-1) = (f(:,3:n) - f(:,1:n-2)) ./ h;
    end
    varargout{2} = g;
    
elseif ndim > 2
    % N-D case
    for k = 2:ndim
        n = siz(k);
        newsiz = [prod(siz(1:k-1)) siz(k) prod(siz(k+1:end))];
        nf = reshape(f,newsiz);
        h = reshape(loc{k},1,[]);
        g = zeros(size(nf), 'like', prototype);
        
        % Take forward differences on left and right edges
        if n > 1
            g(:,1,:) = (nf(:,2,:) - nf(:,1,:))/(h(2)-h(1));
            g(:,n,:) = (nf(:,n,:) - nf(:,n-1,:))/(h(end)-h(end-1));
        end
        
        % Take centered differences on interior points
        if n > 2
            h = h(3:n) - h(1:n-2);
            g(:,2:n-1,:) = (nf(:,3:n,:) - nf(:,1:n-2,:)) ./ h;
        end
        
        varargout{k} = reshape(g,siz);
    end
end

% Swap 1 and 2 since x is the second dimension and y is the first.
if ndim > 1
    varargout([2 1]) = varargout([1 2]);
elseif rflag
    varargout{1} = varargout{1}.';
end


%-------------------------------------------------------
function [f,ndim,loc,rowflag] = parse_inputs(f,h)

loc = {}; % spacing along the x,y,z,... directions
ndimsf = ndims(f);
ndim = ndimsf;
rowflag = false;
if isvector(f)
    ndim = 1;
    if isrow(f) % Treat row vector as a column vector
        rowflag = true;
        f = f.';
    end
end

% Default step sizes: hx = hy = hz = 1
indx = size(f);
if isempty(h)
    % gradient(f)
    loc = cell(1,ndimsf);
    for k = 1:ndimsf
        loc(k) = {1:indx(k)};
    end
elseif isscalar(h) % gradient(f,h)
    if isscalar(h{1})
        % Expand scalar step size
        loc = cell(1,ndimsf);
        for k = 1:ndimsf
            loc(k) = {h{1}*(1:indx(k))};
        end
    elseif ndim == 1
        % Check for vector case
        if numel(h{1}) ~= numel(f)
            error(message('MATLAB:gradient:InvalidGridSpacing'));
        end
        loc(1) = h(1);
    else
        error(message('MATLAB:gradient:InvalidInputs'));
    end
elseif ndimsf == numel(h)  % gradient(f,hx,hy,hz,...)
    % Swap 1 and 2 since x is the second dimension and y is the first.
    loc = h;
    if ndim > 1
        loc([2 1]) = loc([1 2]);
    end
    % replace any scalar step-size with corresponding position vector, and
    % check that the values specified in each position vector is the right
    % shape and size
    for k = 1:ndimsf
        if isscalar(loc{k})
            loc{k} = loc{k}*(1:indx(k));
        elseif ~isvector(squeeze(loc{k})) || numel(loc{k}) ~= size(f, k)
            error(message('MATLAB:gradient:InvalidGridSpacing'));
        end
    end 
else
    error(message('MATLAB:gradient:InvalidInputs'));

end
