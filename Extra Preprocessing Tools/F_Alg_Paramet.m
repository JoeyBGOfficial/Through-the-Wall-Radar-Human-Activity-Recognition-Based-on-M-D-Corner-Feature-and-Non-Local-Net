function [CalcVal, f] = F_Alg_Paramet(Signal,Signal_ph,DiffParamets,graphs,wname,wlevels)
global Fs

ord = DiffParamets{1};
frqrange = DiffParamets{2};
if strcmp(wname, 'pmusic')
    [CalcVal, f] = pmusic(Signal,ord,[],Fs, frqrange);
elseif strcmp(wname, 'peig')
    [CalcVal, f] = peig(Signal, ord, 512, Fs, frqrange);
elseif strcmp(wname, 'rootmusic')
    [CalcVal, f] = rootmusic(Signal, ord, frqrange, Fs);
elseif strcmp(wname, 'pburg')
    [CalcVal, f] = pburg(Signal, ord, 512, Fs, frqrange);
elseif strcmp(wname, 'pcov')
    [CalcVal, f] = pcov(Signal,ord,512,Fs, frqrange);
elseif strcmp(wname, 'pyulear')
    [CalcVal, f] = pyulear(Signal,ord,512,Fs, frqrange);
else

    
    error('Wrong type');
end
    
end