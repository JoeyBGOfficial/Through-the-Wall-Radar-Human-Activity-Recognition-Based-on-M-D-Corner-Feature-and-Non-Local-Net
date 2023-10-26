function [CalcVal,a2,ind] = F_Alg_HHTperiodic(Signal,SignalF,graphs,TypeSignal)
global CalcValHHTperiodic cfarWin f1 HHTLevelBin Alarms Periodic leveldef Fcut
% Входные параметры
if TypeSignal==1
elseif TypeSignal==2
    threshHHT=Alarms(1,4);       % Постоянное значение обнаружителя
end



% Алгоритм
[imfinseHHT,imfinsfHHT,ind,CalcVal,E,F,a2] = UnmvDetectHHT(Signal,Periodic);


% Обнаружитель
if TypeSignal==1    % Если вычисляем статистику
    A=ones(1,a2);
    A(CalcVal(:,2)<Fcut(1,1) | CalcVal(:,2)>Fcut(1,2))=0;
    CalcVal(:,1:2)=CalcVal(:,1:2).*A';      % Обнуление частот выше заданного значения
    DetectionSign(1,1)=max(CalcVal(:,1));   % Запись максимального числа в массив
elseif TypeSignal==2    % Если принимаем решение по сигналу дальности
    if all(leveldef==1)
        HHTLevel=HHTLevelBin(1,:).*threshHHT;    % присвоение значений threshHHT пороговой функции HHTLevel
    elseif all(leveldef==2) % Не реализованно
        threshHHT=max(CalcVal(:,1).*HHTLevelBin(2,ind)');      % Вычисление threshHHT из спектра CalcVal
        HHTLevel=HHTLevelBin(1,:).*threshHHT;    % присвоение значений
    else
    end
    Detector=CalcVal(:,1)>HHTLevel(ind)';   % Сравнение спектра CalcVal с пороговой функции HHTLevel в ind
    Detector=Detector.*HHTLevelBin(1,ind)'; % Обнуление заданных частот
    % Вывод
    DetectionSign=any(Detector);    % Детектирование
    DetectedSpectrum=zeros(1,length(f1));
    DetectedSpectrum(ind)=CalcVal(:,1).*Detector; % Спектр превысивший пороговый уровнь
end
CalcValHHTperiodic=CalcVal;

%Гарфика
if graphs==1
    Graphic(imfinseHHT,imfinsfHHT,Detector,SignalF,f1,HHTLevel,Signal,ind,CalcVal);
    Graphic2(Detector,Signal,SignalF,f1,HHTLevel,ind,CalcVal);
    pause;
elseif graphs==2
    Graphic2(Detector,Signal,SignalF,f1,HHTLevel,ind,CalcVal);
elseif graphs==0
elseif graphs==3
else
    error('Incorrect graphs Input value');
end

end



%----------------------%
%----------------------%
%----------------------%

% Алгоритм
function [imfinseHHT,imfinsfHHT,ind,CalcVal,E,F,a2] = UnmvDetectHHT(Signal,Periodic)
global Fs f1
            %% Декомпозиция
            [imf,residual(:),info] = emd(Signal,'Interpolation','spline','Display',0,'MaxNumIMF',5);
            [~,a2]=size(imf);
            Imf(:,1:a2)=imf;
            %% Вычисление мгновенных частот и амплитуд
            [imfinseHHT,imfinsfHHT] = HHT(Imf,Fs,a2,numel(Signal));
            %% Уменьшение неопределенности на краях
            nKUnS=10;                            % Количество кадров неопределенности
            imfinseHHT=imfinseHHT(nKUnS:end-nKUnS,:);
            imfinsfHHT=imfinsfHHT(nKUnS:end-nKUnS,:);
            %% Вычисление характеристик мод
            CalcVal=zeros(a2,10);
            CalcVal(:,1)=sum(imfinseHHT);       % накопление энергий мод
%             CalcVal(:,1)=abs(sum(imfinseHHT));       % накопление энергий мод
            CalcVal(:,2)=mean(imfinsfHHT);      % вычисление средней частоты моды
%             CalcVal(:,2)=median(imfinsfHHT);      % вычисление средней частоты моды
            %% Найти периоды анализируемых мод с характеристиками удовлетворящими условия
            [newnK,~]=size(imfinseHHT);
            [Cwin,Nwin] = WinFunc(Periodic(1),Periodic(2),newnK);       % Создание преиодов окон
            E=zeros(Nwin,a2);                       % Массивы для вычислений
            F=zeros(Nwin,a2);                       % Массивы для вычислений
            E1max=zeros(1,a2);
%             F1min=zeros(1,a2);
            BoundDetection=zeros(a2,2);             % Приод найденных окон
            imfinseHHT2=zeros(Periodic(1)+1,a2);
            imfinsfHHT2=zeros(Periodic(1)+1,a2);
            for ik=1:a2
                for i=1:Nwin
                    E(i,ik)=sum(imfinseHHT(Cwin(1,i):Cwin(2,i),ik));    % подсчет энергии в окне
                    F(i,ik)=std(imfinsfHHT(Cwin(1,i):Cwin(2,i),ik));    % подсчет std частоты в окне
                end
                [E1max(1,ik),~]=find(E(:,ik)==max(E(:,ik)));
                BoundDetection(ik,:)=[Cwin(1,E1max(1,ik)),Cwin(2,E1max(1,ik))];
%                 [F1min(1,ik),~]=find(F(:,ik)==max(F(:,ik)));
%                 BoundDetection(ik,:)=[Cwin(1,F1min(1,ik)),Cwin(2,F1min(1,ik))];
                imfinseHHT2(:,ik)=imfinseHHT(BoundDetection(ik,1):BoundDetection(ik,2),ik);
                imfinsfHHT2(:,ik)=imfinsfHHT(BoundDetection(ik,1):BoundDetection(ik,2),ik);
            end
            imfinseHHT=imfinseHHT2;
            imfinsfHHT=imfinsfHHT2;
            %% Привязка к сетке частот
            A=abs(f1-CalcVal(:,2)');     % Округление значений частоты до ближайшего по сетке частот
            [ind,~]=find(A==min(A));    % Определение индексов частот
            CalcVal(:,2)=f1(ind);        % Замена частот на сетку
            %% Нормировка амплитудных значений
%             CalcVal(:,1)=CalcVal(:,1)/nK;
                [newnK,~]=size(imfinseHHT);
                % CalcVal(:,1)=CalcVal(:,1)/newnK;
            clear Imf  
end

%HHT Hilbert Spectrum
function [imfinseHHT,imfinsfHHT] = HHT(Imf,fs,a2,nK)
imfinsfHHT = zeros(nK,a2);
imfinseHHT = zeros(nK,a2);
for i1 = 1:a2
    sig = hilbert(Imf(:,i1));
    energy = abs(sig).^2;
    phaseAngle = angle(sig);
    omega = gradient(unwrap(phaseAngle));
    omega = fs/(2*pi)*omega;
    imfinsfHHT(:,i1) = omega;
    imfinseHHT(:,i1) = energy;
end
end

%Window Function Calculation
function [Cwin,Nwin] = WinFunc(k,thru,nK)
Awin=1:thru:nK;
Bwin=Awin+k;
Bwinlim=find(Bwin>nK);
Awin=Awin(1:Bwinlim(1)-1);
Bwin=Bwin(1:Bwinlim(1)-1);
Cwin=[Awin;Bwin];   % Массив окон
[~,Nwin]=size(Cwin);
end

% Графика imf inE при первышении порогового уровня
function Graphic(imfinseHHT,imfinsfHHT,Detector,SignalF,f1,HHTLevel,Signal,ind,CalcVal)
global nK dt AxTFrames
    AmpsFArray=zeros(1,length(f1));
    AmpsFArray(ind)=CalcVal(:,1);
    TargFreqInd=ind.*Detector;
    TargFreqInd(TargFreqInd==0)=[];
        subplot(513)
    stem(f1,AmpsFArray);
    hold on
    plot(f1,HHTLevel,'g--');
    plot(f1,AmpsFArray,'g*','MarkerIndices',TargFreqInd)
    hold off
        subplot(527)
    waterfall(imfinseHHT')
    title('inE')
        subplot(528)
    waterfall(imfinsfHHT')
    title('inF')
            nT8 = round(numel(SignalF)/2); 
            dfs = 1/(dt*numel(SignalF));        
            F   = 0:dfs:(numel(SignalF)-1)*dfs; % массив сетки частот для ДПФ
            FX = abs(fft(SignalF-mean(SignalF)))/numel(SignalF);  % ДПФ процесса дыхания
        subplot(511); plot(AxTFrames(1:numel(SignalF)),Signal(1:numel(SignalF)),AxTFrames(1:numel(SignalF)),SignalF); grid on;
        title({'Model HHTPeriodic Algorithm';...
                'Сигнал дыхания'});         xlabel('Сек');
        legend('Noised','Filtered Signal');
        subplot(512); plot(F(1:nT8),FX(1:nT8)); grid on;
        title('Спектр сигнала дыхания'); xlabel('Гц');
end

function Graphic2(Detector,Signal,SignalF,f1,HHTLevel,ind,CalcVal)
global axSignal axFilteredSignal axFreqHHTperiodic axThreshHHTperiodic axHHTAmpsperiodic axHHTFreqIndperiodic
    AmpsFArray=zeros(1,length(f1));
    AmpsFArray(ind)=CalcVal(:,1);
    TargFreqInd=ind.*Detector;
    TargFreqInd(TargFreqInd==0)=[];
    axSignal=Signal;
    axFilteredSignal=SignalF;
    axFreqHHTperiodic=f1;
    axThreshHHTperiodic=HHTLevel;
    axHHTAmpsperiodic=AmpsFArray;
    axHHTFreqIndperiodic=TargFreqInd;
end