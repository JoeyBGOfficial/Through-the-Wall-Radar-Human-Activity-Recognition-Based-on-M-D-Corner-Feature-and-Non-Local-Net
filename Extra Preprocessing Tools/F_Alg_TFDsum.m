function [CalcVal] = F_Alg_TFDsum(Signal)
    global nK   
    % ЧПК
    num=4;  % сдвиг кадров
    CalcVal=zeros(nK,1);
    CalcVal(1:nK-num,1)=abs(Signal(1+num:nK,1)-Signal(1:nK-num,1));
    CalcVal=CalcVal.^2; %ЧПК энергия
    CalcVal = sum(CalcVal);
end