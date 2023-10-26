%% A Program for EMD Processing
% Former Author: Team of MATLAB Signal Processing Toolbox;
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
% 
% Citation:
% [1] P. Cao, W. Xia, M. Ye, J. Zhang, and J. Zhou, “Radar‐ID: human 
% identification based on radar micro‐Doppler signatures using deep 
% convolutional neural networks,” IET Radar, Sonar & Navigation, vol. 12, 
% no. 7, pp. 729–734, Jul. 2018.
% [2] M. B.R. and P. Rema, “A Performance based comparative study on the 
% Modified version of Empirical Mode Decomposition with traditional Empirical 
% Mode Decomposition,” Procedia Computer Science, vol. 171, pp. 2469–2475, 2020.

%% EMD Tool
function [varargout] = emd(x,varargin)
% Empirical mode decomposition. 

% #codegen.
signalwavelet.internal.licenseCheck;

narginchk(1,15);
if coder.target('MATLAB') % for matlab
    nargoutchk(0,3);
else
    nargoutchk(1,3);
end

[x,t,td,isTT,opt] = parseAndValidateInputs(x, varargin{:});
[IMF, residual, info] = localEMD(x,t,opt);

if(isTT)
    t = td;
end

if (nargout == 0 && coder.target('MATLAB'))
    signalwavelet.internal.convenienceplot.imfPlot(x, IMF, residual, t, 'emd');
end

if(isTT && coder.target('MATLAB'))
    IMF = array2timetable(IMF,'RowTimes',t);
    residual = array2timetable(residual,'RowTimes',t);
end

if nargout > 0
    varargout{1} = IMF;
end

if nargout > 1
    varargout{2} = residual;
end

if nargout > 2
    varargout{3} = info;
end

end

%--------------------------------------------------------------------------
function [x,t,td,isTT,opt] = parseAndValidateInputs(x,  varargin)
% input type checking
isTT = isa(x,'timetable');
if ~isTT
    % Keep the timetable check so that the error message lists all valid
    % data types. 
    validateattributes(x, {'single','double','timetable'},{'vector'},'emd','X');
else
    coder.internal.errorIf(~coder.target('MATLAB'),'shared_signalwavelet:emd:general:TimetableNotSupportedCodegen')
end

% handle timetable and single
if(isTT)
    signalwavelet.internal.util.utilValidateattributesTimetable(x, {'sorted','singlechannel'});
    [x, t, td] = signalwavelet.internal.util.utilParseTimetable(x);
else
    isSingle = isa(x,'single');
    td = [];
    if(isSingle)
        t = single(1:length(x))';
    else
        t = (1:length(x))';
    end
end

% data integrity checking
validateattributes(x, {'single','double'},{'nonnan','finite','real','nonsparse'},'emd','X');
validateattributes(t, {'single','double'},{'nonnan','finite','real'},'emd','T');

% turn x into column vector
if isrow(x)
    x = x(:);
end

% parse and validate name-value pairs
if(isempty(varargin))
    opt = signalwavelet.internal.emd.emdOptions();
else
    opt = signalwavelet.internal.emd.emdOptions(varargin{:});
end

validatestring(opt.Interpolation,{'spline', 'pchip'}, 'emd', 'Interpolation');
validateattributes(opt.SiftStopCriterion.SiftMaxIterations,...
    {'numeric'},{'nonnan','finite','scalar','>',0,'integer'}, 'emd', 'SiftMaxIterations');
validateattributes(opt.SiftStopCriterion.SiftRelativeTolerance,...
    {'numeric'},{'nonnan','finite','scalar','>=',0,'<',1},'emd', 'SiftRelativeTolerance');
validateattributes(opt.DecompStopCriterion.MaxEnergyRatio,...
    {'numeric'},{'nonnan','finite','scalar'}, 'emd', 'MaxEnergyRatio');
validateattributes(opt.DecompStopCriterion.MaxNumExtrema,...
    {'numeric'},{'nonnan','finite','scalar','>=',0,'integer'},'emd','MaxNumExtrema');
validateattributes(opt.DecompStopCriterion.MaxNumIMF,...
    {'numeric'},{'nonnan','finite','scalar','>',0,'integer'},'emd', 'MaxNumIMF');
end

%--------------------------------------------------------------------------
function [IMFs, rsig, info] = localEMD(x, t, opt)
isInMATLAB = coder.target('MATLAB');
isSingle = isa(x,'single');

% get name-value pairs
Interpolation = opt.Interpolation;
MaxEnergyRatio = opt.DecompStopCriterion.MaxEnergyRatio;
MaxNumExtrema = opt.DecompStopCriterion.MaxNumExtrema;
MaxNumIMF = opt.DecompStopCriterion.MaxNumIMF;
SiftMaxIterations = opt.SiftStopCriterion.SiftMaxIterations;
SiftRelativeTolerance = opt.SiftStopCriterion.SiftRelativeTolerance;
Display = opt.Display;

% initialization
rsig = x;
N = length(x);

if(isSingle)
    ArrayType = 'single';
else
    ArrayType = 'double';
end

IMFs = zeros(N, MaxNumIMF, ArrayType);
info.NumIMF = zeros(MaxNumIMF, 1, ArrayType);
info.NumExtrema = zeros(MaxNumIMF, 1, ArrayType);
info.NumZerocrossing = zeros(MaxNumIMF, 1, ArrayType);
info.NumSifting = zeros(MaxNumIMF, 1, ArrayType);
info.MeanEnvelopeEnergy = zeros(MaxNumIMF, 1, ArrayType);
info.RelativeTolerance = zeros(MaxNumIMF, 1, ArrayType);

% preallocate memory
rsigPrev = zeros(N, 1, ArrayType);
mVal = zeros(N, 1, ArrayType);
upperEnvelope = zeros(N, 1, ArrayType);
lowerEnvelope = zeros(N, 1, ArrayType);

% Define intermediate print formats
if(isInMATLAB && opt.Display)
    fprintf('Current IMF  |  #Sift Iter  |  Relative Tol  |  Stop Criterion Hit  \n');
    formatstr = '  %5.0f      |    %5.0f     | %12.5g   |  %s\n';
end

% use different functions under different environment
if(isInMATLAB)
    if(~isSingle)
        localFindExtramaIdx = @(x) signalwavelet.internal.emd.cg_utilFindExtremaIdxmex_double(x);
    else
        localFindExtramaIdx = @(x) signalwavelet.internal.emd.cg_utilFindExtremaIdxmex_single(x);
    end
else
    localFindExtramaIdx = @(x) signalwavelet.internal.emd.utilFindExtremaIdx(x);
end

% extract IMFs
i = 0;
outerLoopExitFlag = 0;
while(i<MaxNumIMF)
    % convergence checking
    [peaksIdx, bottomsIdx] = localFindExtramaIdx(rsig);
    numResidExtrema = length(peaksIdx) + length(bottomsIdx);
    energyRatio = 10*log10(norm(x,2)/norm(rsig,2));
    
    if energyRatio > MaxEnergyRatio
        outerLoopExitFlag = 1;
        break
    end
    
    if numResidExtrema < MaxNumExtrema
        outerLoopExitFlag = 2;
        break
    end
    
    % SIFTING process initialization
    rsigL = rsig;
    rtol = ones(1, ArrayType);
    k = 0;
    SiftStopCriterionHit = 'SiftMaxIteration';
    
    % Sifting process
    while (k<SiftMaxIterations)
        % check convergence
        if(rtol<SiftRelativeTolerance)
            SiftStopCriterionHit = 'SiftMaxRelativeTolerance';
            break;
        end
        
        % store previous residual
        rsigPrev(1:N) = rsigL;
        
        % finding peaks
        [peaksIdx, bottomsIdx] = localFindExtramaIdx(rsigL);
        
        if((length(peaksIdx) + length(bottomsIdx))>0)
            % compute upper and lower envelope using extremas
            [uLoc, uVal, bLoc, bVal] = computeSupport(t, rsigL, peaksIdx, bottomsIdx);
            upperEnvelope(:) = interp1(uLoc, uVal, t, Interpolation);
            lowerEnvelope(:) = interp1(bLoc, bVal, t, Interpolation);
            
            % subtract mean envelope from residual
            mVal(1:N) = (upperEnvelope + lowerEnvelope)/2;
        else
            mVal(1:N) = 0;
        end
        
        rsigL = rsigL - mVal;
        
        % residual tolerance
        rtol = (norm(rsigPrev-rsigL,2)/norm(rsigPrev,2))^2;
        k = k + 1;
    end
    
    if(isInMATLAB && Display)
        fprintf(formatstr, i+1, k, rtol, SiftStopCriterionHit);
    end
    
    % record information
    [peaksIdx, bottomsIdx] = localFindExtramaIdx(rsigL);
    numZerocrossing = sum(diff(sign(rsigL))~=0);
    info.NumIMF(i+1) = i+1;
    info.NumExtrema(i+1) = length(peaksIdx) + length(bottomsIdx);
    info.NumZerocrossing(i+1) = numZerocrossing;
    info.MeanEnvelopeEnergy(i+1) = mean(mVal.^2);
    info.NumSifting(i+1) = k;
    info.RelativeTolerance(i+1) = rtol;
    
    % extract new IMF and subtract the IMF from residual signal
    IMFs(:,i+1) = rsigL;
    rsig = rsig - IMFs(:,i+1);
    i = i + 1;
end

if(isInMATLAB && Display)
    switch outerLoopExitFlag
        case 0
            disp(getString(message('shared_signalwavelet:emd:general:MaxNumIMFHit')));
        case 1
            disp(getString(message('shared_signalwavelet:emd:general:MaxEnergyRatioHit', 'MaxEnergyRatio')));
        case 2
            disp(getString(message('shared_signalwavelet:emd:general:MaxNumExtremaHit', 'MaxNumExtrema')));
    end
end

% remove extra portion
IMFs = IMFs(:,1:i);
info.NumIMF = info.NumIMF(1:i);
info.NumExtrema = info.NumExtrema(1:i);
info.NumZerocrossing = info.NumZerocrossing(1:i);
info.NumSifting = info.NumSifting(1:i);
info.MeanEnvelopeEnergy = info.MeanEnvelopeEnergy(1:i);
info.RelativeTolerance = info.RelativeTolerance(1:i);
end

%--------------------------------------------------------------------------
function [uLoc, uVal, bLoc, bVal] = computeSupport(t, rsigL, pksIdx, btmsIdx)
% compute support for upper and lower envelope given input signal rsigL
N = length(t);
if(isempty(pksIdx))
    pksIdx = [1; N];
end

if(isempty(btmsIdx))
    btmsIdx = [1; N];
end

pksLoc = t(pksIdx);
btmsLoc = t(btmsIdx);

% compute envelop for wave method
% extended waves on the left
[lpksLoc, lpksVal, lbtmLoc, lbtmVal] = signalwavelet.internal.emd.emdWaveExtension(t(1), rsigL(1),...
    pksLoc(1), rsigL(pksIdx(1)),...
    btmsLoc(1), rsigL(btmsIdx(1)),...
    -1);

% extended waves on the right
[rpksLoc, rpksVal, rbtmLoc, rbtmVal] = signalwavelet.internal.emd.emdWaveExtension(t(end), rsigL(end),...
    pksLoc(end), rsigL(pksIdx(end)),...
    btmsLoc(end), rsigL(btmsIdx(end)),...
    1);

% append extended wave to extrema
uLoc = [lpksLoc;pksLoc;rpksLoc];
uVal = [lpksVal;rsigL(pksIdx);rpksVal];
bLoc = [lbtmLoc;btmsLoc;rbtmLoc];
bVal = [lbtmVal;rsigL(btmsIdx);rbtmVal];
end
