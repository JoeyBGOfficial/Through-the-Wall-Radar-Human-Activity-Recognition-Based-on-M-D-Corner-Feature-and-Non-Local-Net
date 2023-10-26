function [CalcVal, f] = F_Alg_WT(Signal, alg_type, wname, wlev, DiffParamets)
global Fs f1 nK

% Алгоритм
% Декомпозиция
[imf]=F_WavDec(wlev, wname, Signal);  % Вейвлет декомпозиция
% Вычисление мгновенных частот и амплитуд
[imfinseHHT, imfinsfHHT] = F_HHT(imf, Fs);
% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
[CalcVal, imfinseHHT, imfinsfHHT] = F_inmods_vals(imfinseHHT, imfinsfHHT);
ind = CalcVal(:,3);


if strcmp(alg_type, 'WT')
    f = f1;
elseif strcmp(alg_type, 'WTold')
    % Выч производной по частоте
    [~, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 1);
    % Сложение оставшихся мод
    % imf = hilbert (imf);
    sumedimf = sum(imf,2);
    %FFT для обнаружения
    Spec = fft(sumedimf);
    PSpec = abs(Spec/nK);
    absf = PSpec(1:nK/2+1);
    absf(2:end-1) = 2*absf(2:end-1);
    CalcVal = absf;
    f = Fs*(0:(nK/2))/nK;
elseif strcmp(alg_type, 'WTthr')
    % надо сделать чтобы моды складывались если значение энергии первышает
    % рассчитанное на количество отсчетов
    % можно сделать условие внутри F_ImfExe
    % Выч производной по частоте и фильтрация
    [CalcVal, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 1);
    f = f1;
elseif strcmp(alg_type, 'WTthr2')
    % Выч производной по частоте и фильтрация, обьединение мод
    [CalcVal, imf] = F_ImfExe(CalcVal, DiffParamets, imf, imfinsfHHT, imfinseHHT, 2);
    f = f1;
else
    error('Wrong parametr');
end

% Hilbert_Spectrum
% figure('Position',[700 400 400 200],'Units','pixels');
% hht(imf,Fs);

end