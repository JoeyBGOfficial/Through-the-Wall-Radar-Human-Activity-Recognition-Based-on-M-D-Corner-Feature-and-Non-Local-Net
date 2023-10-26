function [CalcVal, f] = F_Alg_WssT(Signal, alg_type)
global nK Fs f2 fwsst

%Алгоритм
opts = {'Window',kaiser(10),'OverlapLength',8,'FFTLength', 2*(numel(f2)-1)};
[Spectrum, f] = wsst(Signal,Fs);

% Спектр
CalcVal=abs(Spectrum);

end