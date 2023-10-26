function [CalcVal, f] = F_Alg_FFT(Signal, alg_type)
global nK Fs

if strcmp(alg_type, 'FFT')
elseif strcmp(alg_type, 'HTFFT')
    Signal = hilbert(Signal);
end

%Алгоритм
Spec = fft(Signal);
PSpec = abs(Spec/nK);
absf = PSpec(1:nK/2+1);
absf(2:end-1) = 2*absf(2:end-1);
CalcVal=absf;

f = Fs*(0:(nK/2))/nK;

end
