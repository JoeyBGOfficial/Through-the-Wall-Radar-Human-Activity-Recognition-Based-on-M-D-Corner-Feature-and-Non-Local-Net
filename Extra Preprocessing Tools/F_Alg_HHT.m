function [CalcVal, f] = F_Alg_HHT(Signal, alg_type, DiffParamets)
global Fs f1 nK

% Алгоритм
%% Декомпозиция
[imf,~,~] = emd(Signal,'Interpolation','spline','Display',0);
%% Вычисление мгновенных частот и амплитуд
[imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
%% Вычисление характеристик мод Нормировка амплитудных значений
[CalcVal,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);
ind = CalcVal(:,3);


if strcmp(alg_type, 'HHT')
    f = f1;
elseif strcmp(alg_type, 'HHTold')
    % Выч производной по частоте и фильтрация
    [CalcVal, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 1);
    % Сложение оставшихся мод
%     imf(:, index) = [];
    sumedimf = sum(imf,2);
%     imf = hilbert (imf);
    %FFT для обнаружения
    Spec = fft(sumedimf);
    PSpec = abs(Spec/nK);
    absf = PSpec(1:nK/2+1);
    absf(2:end-1) = 2*absf(2:end-1);
    CalcVal=absf;
    f = Fs*(0:(nK/2))/nK;
elseif strcmp(alg_type, 'HHTthr')
    % Выч производной по частоте и фильтрация
    [CalcVal, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 1);
    f = f1;
elseif strcmp(alg_type, 'HHTthr2')
    % Выч производной по частоте и фильтрация, обьединение мод
    [CalcVal, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 2);
    f = f1;
else
    error('Wrong parametr');
end


% Hilbert_Spectrum
% figure('Position',[700 400 400 200],'Units','pixels');
% hht(imf,Fs);

clear Imf               
end
