function [CalcVal, f] = F_Alg_STFFT(Signal, alg_type)
global nK Fs f2

%Алгоритм
opts = {'Window',kaiser(10),'OverlapLength',8,'FFTLength',2*(numel(f2)-1)};
[Spectrum,f,t] = stft(Signal,Fs,opts{:});

% оставить однополосный спектр
Spectrum(f<0,:) = [];   
f = f(f>=0);

% Спектр
CalcVal=abs(Spectrum);

end
