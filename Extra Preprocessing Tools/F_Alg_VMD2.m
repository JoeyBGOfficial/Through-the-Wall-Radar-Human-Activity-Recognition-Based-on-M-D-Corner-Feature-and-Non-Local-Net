function [CalcVal,ind] = F_Alg_VMD2(Signal,DiffParamets,graphs,wname,wlevels)
global Fs
Signal = Signal + 0.01*mean(Signal)*rand(size(Signal));

% vmd(Signal,'Display',1);
[imf,residual] = vmd(Signal,'Display',1);
imf(:,end) = [];

    %% Вычисление мгновенных частот и амплитуд
    [imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
    %% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
    [~,~,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);
    [CalcVal,ind] = F_CalculateValues(imfinseHHT,imfinsfHHT);
    %% Выч производной по частоте и фильтрация
    [CalcVal,index] = F_ImfExe(CalcVal,DiffParamets,imfinsfHHT,imfinseHHT);
    %%
    imf(:,index) = [];
    imf = sum(imf,2);
    plotimfs(imf,residual,Signal,'VMD',[0 500 400 400])
    figure('Position',[0 100 400 400],'Units','pixels');
    hht(imf,Fs)

% emd(Signal,'MaxNumIMF',5);
[imf,residual] = emd(Signal,'SiftRelativeTolerance',00.1,'SiftMaxIterations',500,'Display',1);
      %% Вычисление мгновенных частот и амплитуд
    [imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
    %% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
    [~,~,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);
    [CalcVal,ind] = F_CalculateValues(imfinseHHT,imfinsfHHT);
    %% Выч производной по частоте и фильтрация
    [CalcVal,index] = F_ImfExe(CalcVal,DiffParamets,imfinsfHHT,imfinseHHT);
    %% 
    imf(:,index) = [];
    imf = sum(imf,2);
    plotimfs(imf,residual,Signal,'EMD',[400 500 400 400])
    figure('Position',[400 100 400 400],'Units','pixels');
    hht(imf,Fs)

[imf]=F_WavDec(wlevels,wname,Signal);  % Вейвлет декомпозиция
residual = zeros(1,numel(Signal));
      %% Вычисление мгновенных частот и амплитуд
    [imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
    %% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
    [~,~,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);
    [CalcVal,ind] = F_CalculateValues(imfinseHHT,imfinsfHHT);
    %% Выч производной по частоте и фильтрация
    [CalcVal,index] = F_ImfExe(CalcVal,DiffParamets,imfinsfHHT,imfinseHHT);
    %%
    imf(:,index) = [];
    imf = sum(imf,2);
    plotimfs(imf,residual,Signal,'WD',[800 500 400 400])
    figure('Position',[800 100 400 400],'Units','pixels');
    hht(imf,Fs)

% pause;

% return;
    

CalcVal = 1;
ind = 1;
close all;

end


%----------------------%
%----------------------%
%----------------------%
function plotimfs(imf,residual,Signal,Alg,figpos)
global AxTFrames
[a,b] = size(imf);
ng = (b+2)*2;
nga = b+2;

figure('Position',figpos,'Units','pixels');
subplot(nga,2,1)
plot(AxTFrames,Signal)
xlabel('Время [с]')
ylabel('Signal');
title([Alg 'Showing Imf ' num2str(b) 'of' num2str(b)])

subplot(nga,2,2)
[absf,f] = fftspect(Signal);
plot(f,absf)
xlabel('Time [s]')
ylabel('Signal');
title([Alg 'Spectrum Imf ' num2str(b) 'of' num2str(b)])

for i = 1:b
    ngs = 3:2:ng;
    subplot(nga,2,ngs(i))
    plot(AxTFrames,imf(:,i))
    xlabel('Time [s]')
    ylabel(['Imf ' num2str(i)]);
    
    subplot(nga,2,ngs(i)+1)
    [absf,f] = fftspect(imf(:,i));
    plot(f,absf)
    xlabel('Frequency [Hz]')
    ylabel(['|Imf ' num2str(i) '|(f)']);
end

subplot(nga,1,b+2)
plot(AxTFrames,residual)
xlabel('Время [с]')
ylabel('Resudial');

end

function [absf,f] = fftspect(Signal)
global Fs nK
f = Fs*(0:(nK/2))/nK;
Y = fft(Signal-mean(Signal));
P2 = abs(Y/nK);
absf = P2(1:nK/2+1);
absf(2:end-1) = 2*absf(2:end-1);
end