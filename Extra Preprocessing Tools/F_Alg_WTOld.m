function [CalcVal,ind] = F_Alg_WTOld(Signal,wname,wlevels, DiffParamets)
global Fs nK f2

% Алгоритм
%% Декомпозиция
[imf]=F_WavDec(wlevels,wname,Signal);  % Вейвлет декомпозиция
%% Вычисление мгновенных частот и амплитуд
[imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
%% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
[CalcVal,ind,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);



%% Выч производной по частоте и фильтрация
[CalcVal,index] = F_ImfExe(CalcVal,DiffParamets,imfinsfHHT,imfinseHHT);

%% Сложение оставшихся мод
imf(:,index) = [];
imf = sum(imf,2);
imf = hilbert (imf);
%% FFT для обнаружения
[absf,f] = fftspect(imf);
CalcVal=absf;

% SignalRe=SignalRe-mean(SignalRe);
% SignalRe=hilbert(SignalRe);
% Spectrum = fft(SignalRe);
% AbsSpectrum = abs(Spectrum);         % Нормировка
% AbsSpectrum = AbsSpectrum(1:nK/2+1);    % Половина спектра
% CalcVal=absf;

clear Imf
end

%----------------------%
%----------------------%
%----------------------%
% Dейвлет декомпозиции
function [absf,f] = fftspect(Signal)
global Fs nK
f = Fs*(0:(nK/2))/nK;
Y = fft(Signal-mean(Signal));
P2 = abs(Y/nK);
absf = P2(1:nK/2+1);
absf(2:end-1) = 2*absf(2:end-1);
end