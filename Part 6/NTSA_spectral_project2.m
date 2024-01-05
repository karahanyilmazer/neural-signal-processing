%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Project 3-2: Topography of alpha-theta ratio
% Instructor: sincxpress.com
%
%%

% Extract alpha/theta ratio for each channel in two time windows:
%   [-800 0] and [0 800]. Use trial-specific power. Plot the topographies.
%   theta=3-8 Hz, alpha=8-13 Hz. Zero-pad to NFFT=1000


% load EEG data
load sampleEEGdata.mat
EEG.data = double(EEG.data);

% timing parameters
tidxPre = EEG.times>-800 & EEG.times<0;   % pre-stimulus



% FFT parameters
nfft = 1000; % zero-padding!


% spectral boundary indices
theta = 
alpha = 

%% extract power

% obtain Fourier coefficients and extract power spectrum
dataXpre = 
dataXpst = 

% band-limited power
thetaPre = 
thetaPst = 


% compute ratios
ratPre = 

%% now for plotting

figure(1), clf

% define color limits for topoplots


%%% first in raw units
subplot(231)
topoplotIndie(ratPre,EEG.chanlocs,'electrodes','off','numcontour',0);
title('\alpha/\theta raw PRE')
set(gca,'clim',rawclim)

subplot(232)

subplot(233)


%%% repeat for log scaled


%% done.
