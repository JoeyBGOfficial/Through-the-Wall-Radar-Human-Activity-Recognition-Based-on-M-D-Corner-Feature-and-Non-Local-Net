%% A Program for Nextpow2
% Former Author: Team of MATLAB;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The nextpow2 function in MATLAB is used to calculate the next power of 2
% that is greater than or equal to a given number.
% The nextpow2 function takes a single input argument, which is the number
% for which we want to find the next power of 2. It returns the exponent
% (or power) of 2 that yields the next power of 2.
% Internally, the nextpow2 function performs a logarithmic calculation to
% determine the exponent. It computes the logarithm base 2 of the input number,
% rounds it up to the nearest integer using the ceil function, and then
% returns this value.
% The purpose of finding the next power of 2 is often related to signal
% processing and numerical computations. Many algorithms and techniques,
% such as the Fast Fourier Transform (FFT) algorithm, require the input
% size to be a power of 2 for optimal performance. The nextpow2 function
% helps in determining the appropriate size for signal or data processing
% by ensuring that the size is sufficient for efficient computations.
%
% Citation: None.

%% Function Tool for Nextpow2
function p = nextpow2(n)
% Find next higher power of 2.

if ~isinteger(n)
    [f,p] = log2(abs(n));

    % Check for exact powers of 2.
    k = (f == 0.5);
    p(k) = p(k)-1;

    % Check for infinities and NaNs
    k = ~isfinite(f);
    p(k) = f(k);

else % integer case
    p = zeros(size(n),class(n));
    nabs = abs(n);
    x = bitshift(nabs,-1);
    while any(x, 'all')
        p = p + sign(x);
        x = bitshift(x,-1);
    end
    % Adjust for all non powers of 2
    p = p + max(0,sign(nabs - bitshift(ones('like',p),p)));

end
