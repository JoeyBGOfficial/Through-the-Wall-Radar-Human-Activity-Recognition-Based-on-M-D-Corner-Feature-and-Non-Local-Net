function [CalcVal,ind] = F_Alg_HHTthr(Signal,DiffParamets,graphs)
global Fs f1 nK
% Хотим значения imf обнулить при заданном пороговом уровне
% Рассматривать все на графике mesh, что как разновидность алгоритмовы

% num = 1;  % сдвиг кадров
% SignalMoved = zeros(nK,1);
% SignalMoved(1:nK-num,1) = Signal(1+num:nK,1)-Signal(1:nK-num,1);
% Signal = SignalMoved;

% Алгоритм
%% Декомпозиция
[imf,~,~] = emd(Signal,'Interpolation','spline','Display',0);
%% Вычисление мгновенных частот и амплитуд
[imfinseHHT,imfinsfHHT] = F_HHT(imf,Fs);
%% Уменьшение неопределенности на краях Вычисление характеристик мод Нормировка амплитудных значений
[CalcVal,ind,imfinseHHT,imfinsfHHT] = F_inmods_vals(imfinseHHT,imfinsfHHT);

%     inda = find(imfinseHHT==max(imfinseHHT));
%     CalcVal(:,1) = imfinseHHT(inda);       % накопление энергий мод
%     CalcVal(:,2) = imfinsfHHT(inda);       % накопление энергий мод
%     CalcVal(:,1) = sum(imfinseHHT);      % вычисление средней частоты моды
%     CalcVal(:,2) = mean(imfinsfHHT);      % вычисление средней частоты моды
%     % Замена частот на сетку
%     A=abs(f1-CalcVal(:,2)');        % Округление значений частоты до ближайшего по сетке частот
%     [ind,~]=find(A==min(A));        % Определение индексов частот
%     CalcVal(:,2)=f1(ind);
%     
%     CalcVal(:,7) = var(imfinsfHHT);
%     CalcVal(CalcVal(:,7) < 1,7) = 1;
%     CalcVal([1,2],:) = 0;
%     
%     CalcVal(:,7) = var(imfinsfHHT)+1;
%     CalcVal(:,1) = CalcVal(:,1)./CalcVal(:,7);

     CalcVal(:,1) = var(imfinseHHT);
%      CalcVal(:,1) = var(imf);
%     CalcVal(:,1) = var(imf)./var(imfinsfHHT);

%% Выч производной по частоте и фильтрация
% [CalcVal,index] = F_ImfExe(CalcVal,DiffParamets,imfinsfHHT,imfinseHHT);

%     CalcVal(:,6) = var(imfinseHHT);
%     CalcVal(:,7) = var(imfinsfHHT);
%     index = (CalcVal(:,7) > DiffParamets(1,2)) | (CalcVal(:,7) < DiffParamets(1,1));
%     CalcVal(index,:) = 0;
%     index = CalcVal(:,6) < DiffParamets(1,3);
%     CalcVal(index,:) = 0;
    

%     subplot(311)
%     waterfall(imfinseHHT')
%     title('energy')
%     subplot(312)
%     waterfall(imfinsfHHT')
%     title('freq')
%     subplot(313)
%     plot(Signal)
%     title('Signal')

    %% Исключение значений спектра без отстчетов
%     [CalcVal,ind] = F_inmods_vals(imfinseHHT,imfinsfHHT,a2);
%     imfidx = ~isnan(CalcVal(:,1));
%     CalcVal=CalcVal(imfidx,:);
    clear Imf
           
end

%----------------------%
%----------------------%
%----------------------%
function [imfinseHHT,imfinsfHHT] = F_thres_HHTthr(imf,Fs,a2,imfinseHHT,imfinsfHHT)
global AxTFrames
% построить исходный hht
close all
figure('Position',[800 100 400 400],'Units','pixels');
hht(imf,Fs);

figure('Position',[800 500 400 400],'Units','pixels');
subplot(211)
waterfall(imfinseHHT')
title('inE')
    subplot(212)
waterfall(imfinsfHHT')
title('inF')

% отфильтровать
thr = 0.1 * max(max(imfinseHHT));
indx = ones(size(imfinseHHT));
indx(imfinseHHT<thr)=NaN;

imfinseHHT=imfinseHHT.*indx;
imfinsfHHT=imfinsfHHT.*indx;

% построить hht
figure('Position',[0 500 400 400],'Units','pixels');
subplot(211)
waterfall(imfinseHHT')
title('inE')
    subplot(212)
waterfall(imfinsfHHT')
title('inF')

figure('Position',[0 100 400 400],'Units','pixels');
tvec = AxTFrames';
nKUnS=5;                            % Количество кадров неопределенности
tvec=tvec(nKUnS:end-nKUnS,:);
for i = 1:a2
    % Plot each IMF
    insfi = imfinsfHHT(:,i);
    insei = imfinseHHT(:,i);
        patch([tvec(1);tvec;tvec(end)], [0;insfi;0], [nan;insei;nan], ...
            'EdgeColor','interp',...
            'FaceColor', 'none', 'FaceVertexAlphaData',[nan;insei;nan],...
            'LineWidth', 2, 'FaceAlpha', 'interp');
end
% xyrange = [tvec(1),tvec(end),FRange(1),FRange(2)];
% axis(xyrange);
xlabel('Time')
xlabel('Hz')
title('HHT spectrum thresholded');
colormap parula
colorbar

% обьеденить imf
% Imf=imf.*indx;

% построить спектр

close all
end