%% A Quick Look of Range-Time or Spectrograms of SimHSet
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-1-6;
% Language & Platform: MATLAB R2023a.
%
% Introduction: Here we provide a series of data reading codes, including
% data formats from images to Mat signal files, most of which come from
% open source software or databases. Please refer to the original author
% when using.
% 
% Citation: None.

%% Preparations for Matlab Script
clc;
clear all;
close all;

%% GUI Window for Input
% In this section, we generate a input dialog GUI window for string input.
prompt = 'Which Dataset do you want to access? A. DIAT_JPG, B. DIAT_MAT, C. CI4R';
dlgtitle = 'Choose Dataset';
numlines = [1,81];
defAns = {'CI4R'};
Dlg_Input = inputdlg(prompt,dlgtitle,numlines,defAns);
Input_Dataset = Dlg_Input{1};
Now_Path = pwd;

% Define whether dataset to access.
if strcmp(Input_Dataset,'A') || strcmp(Input_Dataset,'DIAT_JPG') || strcmp(Input_Dataset,'A. DIAT_JPG')
    Dataset_Path = strcat(Now_Path,'\SimHSet_DIAT_JPG');
    cd(Dataset_Path);
    run("Plot_SimHSet_DIAT_JPG.m");
    cd(Now_Path);
elseif strcmp(Input_Dataset,'B') || strcmp(Input_Dataset,'DIAT_MAT') || strcmp(Input_Dataset,'B. DIAT_MAT')
    Dataset_Path = strcat(Now_Path,'\SimHSet_DIAT_MAT');
    cd(Dataset_Path);
    run("Plot_SimHSet_DIAT_MAT.m");
    cd(Now_Path);
elseif strcmp(Input_Dataset,'C') || strcmp(Input_Dataset,'CI4R') || strcmp(Input_Dataset,'C. CI4R')
    Dataset_Path = strcat(Now_Path,'\SimHSet_CI4R');
    cd(Dataset_Path);
    run("Plot_SimHSet_CI4R.m");
    cd(Now_Path);
else
    warndlg('Wrong Input Name, Please Check!','Warning');
end