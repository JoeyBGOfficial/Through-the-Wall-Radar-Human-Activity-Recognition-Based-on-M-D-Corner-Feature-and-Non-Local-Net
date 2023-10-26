function [CalcVal, f] = F_Alg_SsFT(Signal, alg_type)
global nK Fs f2

%Алгоритм
opts = {'Window',kaiser(10),'OverlapLength',8,'FFTLength', 2*(numel(f2)-1)};
[Spectrum,f,t] = fsst(Signal,Fs,'yaxis');

% Спектр
CalcVal=abs(Spectrum);

end
