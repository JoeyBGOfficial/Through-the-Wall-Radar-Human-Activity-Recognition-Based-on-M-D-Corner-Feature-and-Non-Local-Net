function [CalcVal, f, ind] = F_Alg_Cov(Signal,Signals)
    
    [Cov] = cov(Signal,Signals);
    CalcVal= Cov(1,2);

%     [R,P] = corrcoef(Signal,Signals);
%     CalcVal= R(1,2);

    f = [];
    ind = [];
%     CalcVal= max(Pxx);
end