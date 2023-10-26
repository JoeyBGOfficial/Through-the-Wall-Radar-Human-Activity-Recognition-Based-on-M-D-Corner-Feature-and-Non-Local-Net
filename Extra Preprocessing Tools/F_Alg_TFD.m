function [CalcVal, f, ind] = F_Alg_TFD(Signal, diff_params, alg_type)
global nK   

    % ЧПК
    num = diff_params(1);  % сдвиг кадров
    CalcVal = zeros(nK,1);
    CalcVal(1:nK-num,1) = abs(Signal(1+num:nK,1)-Signal(1:nK-num,1));
    CalcVal = CalcVal.^2; %ЧПК энергия
    
    if strcmp(alg_type, 'TFD')

    elseif strcmp(alg_type, 'TFDsum')
        CalcVal = sum(CalcVal);
    end
    f = [];
    ind = [];
    
end