function [CalcVal] = F_Alg_STFT(Signal)
global nK f2

%Алгоритм
Signal=Signal-mean(Signal);          % Минус постоянная составляющая
SignalF2=Signal-mean(Signal);          % Минус постоянная составляющая
% Одноплосный спектр
%             Spectrum = fft(SignalF2);
%             AbsSpectrum = abs(Spectrum/nK);
%             AbsSpectrum = AbsSpectrum(1:nK/2+1);
%             AbsSpectrum(2:end-1) = 2*AbsSpectrum(2:end-1);
% Гильберт спектр
SignalF2=hilbert(SignalF2);
Spectrum = fft(SignalF2);
% Нормализация кол-ва кадров
% AbsSpectrum = abs(Spectrum/nK);
AbsSpectrum = abs(Spectrum);
AbsSpectrum = AbsSpectrum(1:nK/2+1);
%             AbsSpectrum(2:end-1) = AbsSpectrum(2:end-1);
% CalcVal=[AbsSpectrum, f2'];
CalcVal=AbsSpectrum;

end