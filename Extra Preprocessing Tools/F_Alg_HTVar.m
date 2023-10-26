function [CalcVal] = F_Alg_HTVar(Signal)
    % HTVar
    Signal = hilbert(Signal);
    CalcVal= var(abs(Signal));
end