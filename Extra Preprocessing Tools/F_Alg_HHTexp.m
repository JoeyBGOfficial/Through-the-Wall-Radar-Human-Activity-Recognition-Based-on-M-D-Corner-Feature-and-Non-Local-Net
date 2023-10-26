function [CalcVal,a2,ind] = F_Alg_HHTexp(Signal,SignalF,wname,wlevels,Periodic,DiffParamets,leveldef,Fcut,graphs,TypeSignal)
global HHTLevelBin f1 cfarWin CalcValHHTexperemetnal Alarms
% Входные параметры
if TypeSignal==1
elseif TypeSignal==2
    threshHHT=Alarms(1,6);       % Постоянное значение обнаружителя
end

% Алгоритм
[imfinseHHT,imfinsfHHT,ind,CalcVal,a2] = UnmvDetectHHT(Signal,wname,wlevels,Periodic,DiffParamets);

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
% Массив CalcVal с характеристиками мод для просмтора
ColumnNames={'insEnergy','mean(insFrequency)','','','','','std(Freq)','std(diff(Freq))','','max(diff(Freq))'};
CalcValHHTexperemetnal=[ColumnNames;num2cell(CalcVal)];

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
function [imfinseHHT,imfinsfHHT,ind,CalcVal,a2] = UnmvDetectHHT(Signal,wname,wlevels,Periodic,DiffParamets)
global Fs f1
            %% Декомпозиция
%             [imf]=F_WavDec(wlevels,wname,Signal);  % Вейвлет декомпозиция
            [imf,residual(:),info] = emd(Signal,'Interpolation','spline','Display',0,'MaxNumIMF',5);
            [~,a2]=size(imf);
            Imf(:,1:a2)=imf;
            %% Вычисление мгновенных частот и амплитуд
            [imfinseHHT,imfinsfHHT] = HHT(Imf,Fs,a2,numel(Signal));
%             imfinseHHT=Imf;
            %% Уменьшение неопределенности на краях
            nKUnS=5;                            % Количество кадров неопределенности
            imfinseHHT=imfinseHHT(nKUnS:end-nKUnS,:);
            imfinsfHHT=imfinsfHHT(nKUnS:end-nKUnS,:);
            %% Вычисление характеристик мод
            % Исключение значений часто и амлитуд значения которых меньше
            % средней амплитуды
%             assignin('base','imfinseHHT',imfinseHHT);
%             assignin('base','imfinsfHHT',imfinsfHHT);


%             ms=mean(imfinseHHT);
%             imfen=imfinseHHT;
%             imffn=imfinsfHHT;
%             imfen(imfinseHHT<ms)=NaN;
%             imffn(imfinseHHT<ms)=NaN;
            % взять средний амплитудный отсчет из этих
            % или медианный
            % или самый часто повторяемый
            % так же с частными остчетами
            
            
            % другой вариант взять производную по частоте
            % обнулить те остчеты где перепад больше разумного
            ms=1;
            imffdiff=abs(diff(imfinsfHHT));
            imfen=imfinseHHT;
            imffn=imfinsfHHT;
            imfen(imffdiff>ms)=NaN;
            imffn(imffdiff>ms)=NaN;

            CalcVal=zeros(a2,10);
            for i=1:a2
                inE = isnan(imfen(:,i));
                inF = isnan(imffn(:,i));
                vers = sum (inE);
                if vers > 0
                CalcVal(i,1)=mean(~inE);       % накопление энергий мод
                CalcVal(i,2)=mean(~inF);
                else
                CalcVal(i,1)=mean(imfen(:,i));       % накопление энергий мод
                CalcVal(i,2)=mean(imffn(:,i));    
                end
            end
            







%             subplot(211)
%             waterfall(imfen')
%             subplot(212)
%             waterfall(imffn')
%             
%             return
            
%             CalcVal=zeros(a2,10);
%             CalcVal(:,1)=sum(imfinseHHT);       % накопление энергий мод
% %             CalcVal(:,1)=abs(sum(imfinseHHT));       % накопление энергий мод
%             CalcVal(:,2)=mean(imfinsfHHT);      % вычисление средней частоты моды
%             CalcVal(:,2)=median(imfinsfHHT);      % вычисление средней частоты моды
            

            A=abs(f1-CalcVal(:,2)');     % Округление значений частоты до ближайшего по сетке частот
            [ind,~]=find(A==min(A));    % Определение индексов частот
            CalcVal(:,2)=f1(ind);        % Замена частот на сетку
            %% Нормировка амплитудных значений
%             CalcVal(:,1)=CalcVal(:,1)/nK;
%                 [newnK,~]=size(imfinseHHT);
%                 CalcVal(:,1)=CalcVal(:,1)/newnK;
            
                
                
                
            %% Усреднений значений амплитуды по чатотам
%             ZKMotion=CalcVal(:,2);
%             ZKMotion2=1./CalcVal(:,2);
%             ZTimeObserv=1/Fs*nK;
%             ZDelitel=ZTimeObserv./ZKMotion2;
%             ZDelen=CalcVal(:,1)./ZDelitel;
%               ZDelitel2=round(ZDelitel);
%             ZDelen(ZDelen<1)=1;
%             CalcVal(:,1)=CalcVal(:,1)./ZDelitel;
            %% Оценка Std мгновенной частоты
            CalcVal(:,4)=sum(abs(Imf));         % накопление сигнала мод
            CalcVal(:,6)=std(imfinseHHT);       % вычисление std энергии
            CalcVal(:,7)=std(imfinsfHHT);       % вычисление std частоты
            CalcVal(:,9)=var(imfinseHHT);       % вычисление var энергии
            CalcVal(:,10)=var(imfinsfHHT);      % вычисление var частоты
                CalcVal(:,7)=std(imfinsfHHT);           % вычисление std частоты
                CalcVal(:,8)=std(diff(imfinsfHHT));     % мод скорость изменения частоты
                CalcVal(:,10)=max(abs(diff(imfinsfHHT)));
            %% Подавление ВЧ мод с широким спектром
%                 CalcVal(:,7)=var(imfinsfHHT);             % вычисление std частоты
%                 ExeDiff=~(CalcVal(:,7)>1);
%                     CalcVal(:,7)=std(imfinsfHHT);           % вычисление std частоты
%                     CalcVal(:,8)=std(diff(imfinsfHHT));     % мод скорость изменения частоты ы
%                     ExeDiff=~((CalcVal(:,7)>2)|(CalcVal(:,8)>2));
%                 CalcVal=CalcVal(ExeDiff,:);
%                     imfinseHHT=imfinseHHT(:,ExeDiff);
%                     imfinsfHHT=imfinsfHHT(:,ExeDiff);
                %% Найти окном когда сумма максимальна, минимально std freq
%                 k=60;
%                 thru=1;
%                 [~,a2]=(size(Imf));
%                 [Cwin,Nwin] = WinFunc(k,thru,newnK);
%                 E=zeros(Nwin,a2);
%                 F=zeros(Nwin,a2);
%                 BoundDetection=zeros(a2,2);
%                 E1max=zeros(1,a2);
%                 F1min=zeros(1,a2);
%                 imfinseHHT2=zeros(k+1,a2);
%                 imfinsfHHT2=zeros(k+1,a2);
%                 for ik=1:a2
%                     for i=1:Nwin
%                         E(i,ik)=sum(imfinseHHT(Cwin(1,i):Cwin(2,i),ik));    % подсчет энергии в окне
%                         F(i,ik)=std(imfinsfHHT(Cwin(1,i):Cwin(2,i),ik));    % подсчет энергии в окне
%                     end
%                 [E1max(1,ik),~]=find(E(:,ik)==max(E(:,ik)));
%                 BoundDetection(ik,:)=[Cwin(1,E1max(1,ik)),Cwin(2,E1max(1,ik))];
% %                 [F1min(1,ik),~]=find(F(:,ik)==max(F(:,ik)));
% %                 BoundDetection(ik,:)=[Cwin(1,F1min(1,ik)),Cwin(2,F1min(1,ik))];
%                     imfinseHHT2(:,ik)=imfinseHHT(BoundDetection(ik,1):BoundDetection(ik,2),ik);
%                     imfinsfHHT2(:,ik)=imfinsfHHT(BoundDetection(ik,1):BoundDetection(ik,2),ik);
%                 end
%                 imfinseHHT=imfinseHHT2;
%                 imfinsfHHT=imfinsfHHT2;
                %% Подавление ВЧ мод с широким спектром
%                     CalcVal(:,7)=std(imfinsfHHT);           % вычисление std частоты
%                     CalcVal(:,8)=std(diff(imfinsfHHT));     % мод скорость изменения частоты ы
%                     ExeDiff=~((CalcVal(:,7)>2)|(CalcVal(:,8)>2));
%                 CalcVal=CalcVal(ExeDiff,:);
%                     imfinseHHT=imfinseHHT(:,ExeDiff);
%                     imfinsfHHT=imfinsfHHT(:,ExeDiff);
                %% Выч производной по частоте и фильтрация
%                 CalcVal(:,10)=max(abs(diff(imfinsfHHT)));
%                     ExeDiff=~((CalcVal(:,10)>1) | (CalcVal(:,10)<0.001));
%                 CalcVal=CalcVal(ExeDiff,:);
%                     imfinseHHT=imfinseHHT(:,ExeDiff);
%                     imfinsfHHT=imfinsfHHT(:,ExeDiff);
                %% Выделение обнаруженного сигнала
%             IndImf=find(Detector);                      % Номера Imf превысевших пороговую функцию
%             Imfnum=sum(Detector);                       % Количество обнаруженных Imf
%             %% Запись сигналов
%             if Imfnum==0
%                 Targs(:,i)=0;
%                 SX=0;
%             elseif Imfnum==1
%                 Targs(:,i)=Imf(:,IndImf);
%                 SX=1;
%             elseif Imfnum>1
%                 Targs(:,i)=sum(Imf(:,IndImf),2)';
%                 SX=1;
% 
%             end 
%             %% Выборка сигнала по экстремумам
%             if SX==1
% 
%                 CalcValMaxInd=find(CalcVal(:,1)==max(CalcVal(:,1)));
%                 [ymax,imax,ymin,imin] = F_extrema(Targs(:,i));
%                 indx=[imin; imax];
%                 indx=indx(:);
%                 indx=sort(indx,1)';
%                 logic=mod(numel(indx),2) == 0;
%                 if logic
%                     indx=indx(1:end-1);
%                 else                 
%                 end
%                 CorrSignals(1:(indx(end)-indx(1)+1),i)=Targs(indx(1):indx(end),i);
%                 SX=0;               
%             else
%             end
            clear Imf  
end

% Dейвлет декомпозиции
function [DU]=F_WavDec(wlevels,wname,x)
global nK
    %[x1] =wdencmp ('gbl', x, 'db3', 3, 20, 's', 0); %шумоподавление и компрессия
    [C,L]=wavedec(x,wlevels,wname);               %многоуровневый вейвлет анали
        DU=zeros(nK,wlevels);
        for l=1:wlevels         
        DU(:,l)=wrcoef('d',C,L,wname,l);    %Детилизирующий вектор   
        end
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
if isempty(imfinseHHT)
else
        subplot(527)
    waterfall(imfinseHHT')
    title('inE')
        subplot(528)
    waterfall(imfinsfHHT')
end  
    title('inF')
            nT8 = round(numel(SignalF)/2); 
            dfs = 1/(dt*numel(SignalF));        
            F   = 0:dfs:(numel(SignalF)-1)*dfs; % массив сетки частот для ДПФ
            FX = abs(fft(SignalF-mean(SignalF)))/numel(SignalF);  % ДПФ процесса дыхания
        subplot(511); plot(AxTFrames(1:numel(SignalF)),Signal(1:numel(SignalF)),AxTFrames(1:numel(SignalF)),SignalF); grid on;
        title({'Model HHT Experemental Algorithm';...
                'Сигнал дыхания'});         xlabel('Сек');
        legend('Noised','Filtered Signal');
        subplot(512); plot(F(1:nT8),FX(1:nT8)); grid on;
        title('Спектр сигнала дыхания'); xlabel('Гц');
        
        subplot(5,2,9)
        waterfall(abs(diff(imfinseHHT)'))
        title('diff inF')
        subplot(5,2,10)
        waterfall(abs(diff(imfinsfHHT)'))
        title('diff inF')
end

function Graphic2(Detector,Signal,SignalF,f1,HHTLevel,ind,CalcVal)
global axSignal axFilteredSignal axFreqHHTexperemental axThreshHHTexperemental axHHTAmpsexperemental axHHTFreqIndexperemental

    AmpsFArray=zeros(1,length(f1));
    AmpsFArray(ind)=CalcVal(:,1);
    TargFreqInd=ind.*Detector;
    TargFreqInd(TargFreqInd==0)=[];
    axSignal=Signal;
    axFilteredSignal=SignalF;
    axFreqHHTexperemental=f1;
    axThreshHHTexperemental=HHTLevel;
    axHHTAmpsexperemental=AmpsFArray;
    axHHTFreqIndexperemental=TargFreqInd;
end