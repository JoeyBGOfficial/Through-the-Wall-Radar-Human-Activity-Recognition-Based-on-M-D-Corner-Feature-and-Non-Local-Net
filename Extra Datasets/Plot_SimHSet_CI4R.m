%% A Quick Look of Range-Time and Spectrograms of SimHSet
% Former Author: The Laboratory of Computational Intelligence for RADAR;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-1-5;
% Language & Platform: MATLAB R2023a.
%
% Introduction: Human activity dataset from the laboratory of Computational
% intelligence for Radar (Ci4R) has been made public.
% 
% Citation: https://doi.org/10.1117/12.2559155

%% Preparations for Matlab Script
% clc;
% clear all;
close all;

%% Read Radar Data
[FileName,PathName] = uigetfile('*.png','Please select the .png file from SimHSet_CI4R.');
Radar_Image = imread([PathName,FileName]);
Radar_Image_Gray = rgb2gray(Radar_Image);
Radar_Image_Gray = imresize(Radar_Image_Gray,[600 600]);
Radar_Image_Gray = Radar_Image_Gray(12:588,12:588);
Radar_Image_Gray = imresize(Radar_Image_Gray,[600 600]);

%% Display Radar Data
imagesc(Radar_Image_Gray); axis tight;
colormap jet;
colorbar;
xlabel('Time (s)');
ylabel('Doppler (Hz)');
set(gca,'XTick',0:150:600);
set(gca,'XTicklabel',{'0','2','4','6','8'});
set(gca,'YTick',0:100:600);
set(gca,'YTicklabel',{'-120','-80','-40','0','40','80','120'});
set(gca,'FontSize',12);
set(gca,'ydir','normal');
title('Processed DTM','FontWeight','bold','FontSize',14);