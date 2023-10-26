function [CalcVal, f] = F_Alg_1d(Signal, Signals, diff_params, alg_type)
global nK   
    % ЧПК
    if strcmp(alg_type, 'TFD')
        num = diff_params(:);  % сдвиг кадров
        CalcVal = abs(Signal(1+num:nK,1)-Signal(1:nK-num,1));
        CalcVal = CalcVal.^2; %ЧПК энергия
        CalcVal = sum(CalcVal);
    % Корреляция
    elseif strcmp(alg_type, 'Cov')
        [Cov] = cov(Signal,Signals);
        CalcVal= Cov(1,2);
        %     [R,P] = corrcoef(Signal,Signals);
        %     CalcVal= R(1,2);
    % Дисперсия
    elseif strcmp(alg_type, 'Var')
        CalcVal = var(Signal);
    % Дисперсия комплексной огибающей
    elseif strcmp(alg_type, 'HTVar')
        Signal = hilbert(Signal);
        CalcVal= var(abs(Signal));
    end
    f = [];
    
end