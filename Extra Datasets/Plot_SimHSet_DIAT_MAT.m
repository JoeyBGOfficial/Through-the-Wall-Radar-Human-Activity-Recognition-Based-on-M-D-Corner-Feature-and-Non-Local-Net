%% A Quick Look of Range-Time and Spectrograms of SimHSet_DIAT_MAT
% Former Author: Mainak Chakraborty Defence Institute of Advanced
% Technology (DIAT);
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-1-5;
% Language & Platform: MATLAB R2023a.
%
% Introduction: 1. In our dataset, the total number of spectrogram images
% generated using the open-field experiments is 3780, and the class-wise
% details can be found in our journal articles
% 2. The dataset consist of 3780 spectrogram images (Image JPG File (.JPG) and
% .MAT) corresponding to micro-Doppler signatures of different human
% activities; namely (a) army marching, (b) Stone pelting/Grenades
% throwing, (c) jumping with holding a gun, (d) army Jogging, (e) army
% crawling and (f) boxing activities.
% 
% Citation: https://doi.org/10.1016/j.patrec.2022.08.005.

%% Preparations for Matlab Script
% clc;
% clear all;
close all;

%% Read Radar Data
[FileName,PathName] = uigetfile('*.mat','Please select the .mat file from SimHSet_DIAT_MAT.');
Radar_Raw_File = load([PathName,FileName]);
Radar_Data = Radar_Raw_File.data;
Radar_V = Radar_Raw_File.V;
fs = Radar_Raw_File.fs;
ts = Radar_Raw_File.duration;

%% Raw Scalogram Processing of Radar Data
% Parameters.
sampleRate = fs;

% Compute time vector.
t = 0:1/sampleRate:(length(Radar_Data)*1/sampleRate)-1/sampleRate;

% Compute CWT.
% If necessary, substitute workspace variable name for Radar_Data as first
% input to cwt() function in code below.
% Run the function call below without output arguments to plot the results.
[waveletTransform,frequency] = cwt(Radar_Data, sampleRate);
scalogram = abs(waveletTransform);
Scalogram_Normalized = mapminmax(scalogram,0,1);
[Scalogram_Vert,~] = size(scalogram);

%% Plot the Scalogram Result
figure(1);
title('DIAT_MAT Plots');
subplot(1,2,1);
imagesc(flip(Scalogram_Normalized));
axis tight;
colormap parula;
colorbar;
xlabel('Time (s)');
ylabel('Doppler (Hz)');
set(gca,'XTick',0:ts*fs/4:ts*fs);
set(gca,'XTicklabel',{'0',num2str(ts/4),num2str(ts*2/4),num2str(ts*3/4),num2str(ts)});
set(gca,'YTick',0:Scalogram_Vert/6:Scalogram_Vert);
set(gca,'YTicklabel',{'0','20','40','60','80','100','120'});
set(gca,'FontSize',12);
set(gca,'ydir','normal');
title('Raw DTM','FontWeight','bold','FontSize',14);

%% MODWT Processing of Radar Data
% Time-Frequency Calculation.
% Parameters.
frequencyLimits = [0 1]; % Normalized frequency (*pi rad/sample)
voicesPerOctave = 12;
Sampling_Points = fs * ts;

% Decompose signal using the MODWT.
% Pre-Denoising.
Radar_Data_PreDenoised = wdenoise(Radar_Data,12, ...
    Wavelet='sym4', ...
    DenoisingMethod='Bayes', ...
    ThresholdRule='Median', ...
    NoiseEstimate='LevelIndependent');

% Logical array for selecting reconstruction elements.
levelForReconstruction = [false,false,false,false,true];

% Perform the decomposition using modwt.
wt = modwt(Radar_Data_PreDenoised,'sym4',4);

% Construct MRA matrix using modwtmra.
mra = modwtmra(wt,'sym4');

% Sum down the rows of the selected multiresolution signals.
Radar_Data_MODWT = sum(mra(levelForReconstruction,:),1);

% Index the signal time region of interest.
timeLimits = [1,Sampling_Points]; % Sampling.
Radar_Data_ROI = Radar_Data_MODWT(:);
Radar_Data_ROI = Radar_Data_ROI(timeLimits(1):timeLimits(2));

% Convert to cycle sampling.
frequencyLimits = frequencyLimits/2;

% Limit the frequency boundaries of cwt.
frequencyLimits(1) = max(frequencyLimits(1),...
    cwtfreqbounds(numel(Radar_Data_ROI)));

% CWT Calculation.
% Run the function call without output parameters to draw the result.
[WT,F] = cwt(Radar_Data_ROI, ...
    'VoicesPerOctave',voicesPerOctave, ...
    'FrequencyLimits',frequencyLimits);
WTGram = abs(WT);
[WTGram_Vert,~] = size(WTGram);

%% Plot the Time-Frequency Result
subplot(1,2,2);
imagesc(flip(WTGram));
axis tight;
colormap parula;
colorbar;
xlabel('Time (s)');
ylabel('Doppler (Hz)');
set(gca,'XTick',0:ts*fs/4:ts*fs);
set(gca,'XTicklabel',{'0',num2str(ts/4),num2str(ts*2/4),num2str(ts*3/4),num2str(ts)});
set(gca,'YTick',0:WTGram_Vert/6:WTGram_Vert);
set(gca,'YTicklabel',{'0','20','40','60','80','100','120'});
set(gca,'FontSize',12);
set(gca,'ydir','normal');
title('Processed DTM','FontWeight','bold','FontSize',14);