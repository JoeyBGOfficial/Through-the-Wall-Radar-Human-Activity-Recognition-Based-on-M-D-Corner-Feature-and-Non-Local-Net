function [CalcVal, f] = F_Alg_Experiment(Signal, Signal_ph, alg_type, DiffParamets, wname, wlev, graphs)
global Fs

figure();

if strcmp(wname, 'pmusic')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    [CalcVal, f] = pmusic(Signal,ord,[],Fs, frqrange);
        plot(f, 20*log10(abs(CalcVal)));
        xlabel 'Frequency (Hz)', ylabel 'Power (dB)', grid on;
elseif strcmp(wname, 'peig')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    peig(Signal, ord, 512, Fs, frqrange);
elseif strcmp(wname, 'rootmusic')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    rootmusic(Signal, ord, frqrange, Fs);
elseif strcmp(wname, 'pburg')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    pburg(Signal, ord, 512, Fs, frqrange);
elseif strcmp(wname, 'pcov')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    pcov(Signal,ord,512,Fs, frqrange);
elseif strcmp(wname, 'pyulear')
    ord = DiffParamets{1};
    frqrange = DiffParamets{2};
    pyulear(Signal,ord,512,Fs, frqrange);
elseif strcmp(wname, 'pentropy')
    pentropy(Signal,Fs);
    % еще можно от спектра
    % еще от каждой моды
elseif strcmp(wname, 'kurtogram')
    kurtogram(Signal,Fs);
elseif strcmp(wname, 'pkurtosis')
    pkurtosis(Signal,Fs);
elseif strcmp(wname, 'pwelch')
    segmentLength = DiffParamets{1};
    pwelch(Signal,segmentLength,[],512);
elseif strcmp(wname, 'envspectrum')
elseif strcmp(wname, 'xspectrogram')
    window = [];    % kaiser(nwin,30)
    noverlap = [];  % nwin-1
    xspectrogram(Signal,Signal_ph,window,noverlap,512,Fs,'twosided');
elseif strcmp(wname, 'wvd')
    wvd(Signal,Fs);
    wvd(Signal,Fs, 'smoothedPseudo');
elseif strcmp(wname, 'xwvd')
    xwvd(Signal,Signal_ph,Fs);
    xwvd(Signal,Signal_ph,Fs,'smoothedPseudo');
elseif strcmp(wname, 'cpsd')
    cpsd(Signal,Signal_ph,[],[],[],Fs);
elseif strcmp(wname, 'mscohere')
    mscohere(Signal,Signal_ph,[],[],[],Fs);
% elseif strcmp(wname, 'pmtm')   
end


if strcmp(alg_type, 'VMD')

elseif strcmp(wname, 'EEMD')
  
end
    



if isempty(DiffParamets)
    title(sprintf('%s no order', wname));
else
    title(sprintf('%s order-%d', wname, ord));
end

    
end