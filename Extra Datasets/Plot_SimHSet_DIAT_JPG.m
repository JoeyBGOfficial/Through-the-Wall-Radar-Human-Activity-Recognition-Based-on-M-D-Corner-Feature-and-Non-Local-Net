%% A Quick Look of Range-Time and Spectrograms of SimHSet_DIAT_JPG
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
[FileName,PathName] = uigetfile('*.jpg','Please select the .png file from SimHSet_DIAT_JPG.');
Radar_Image = imread([PathName,FileName]);
Radar_Image = Radar_Image(82:931,186:1264,1:3);
Radar_Image_Gray = rgb2gray(Radar_Image);
Radar_Image_Gray = flip(imresize(Radar_Image_Gray,[600 600]));

%% Display Radar Data
imagesc(Radar_Image_Gray); axis tight;
colormap parula;
colorbar;
xlabel('Time (s)');
ylabel('Doppler (Hz)');
set(gca,'XTick',0:150:600);
set(gca,'XTicklabel',{'0','2','4','6','8'});
set(gca,'YTick',0:100:600);
set(gca,'YTicklabel',{'0','20','40','60','80','100','120'});
set(gca,'FontSize',12);
set(gca,'ydir','normal');
title('Processed DTM','FontWeight','bold','FontSize',14);