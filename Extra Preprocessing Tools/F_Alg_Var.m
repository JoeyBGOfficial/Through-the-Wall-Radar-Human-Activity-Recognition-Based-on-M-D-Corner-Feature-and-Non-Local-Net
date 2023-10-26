function [CalcVal, f, ind] = F_Alg_Var(Signal, alg_type)
    % Var
    if strcmp(alg_type, 'Var')
        CalcVal = var(Signal);
    elseif strcmp(alg_type, 'HTVar')
        Signal = hilbert(Signal);
        CalcVal= var(abs(Signal));
    end
    f = [];
    ind = [];
end