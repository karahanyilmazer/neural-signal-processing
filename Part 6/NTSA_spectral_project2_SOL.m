%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Project 3-2: Solutions
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
tidxPst = EEG.times>0    & EEG.times<800; % post-stimulus


% FFT parameters
nfft = 1000; % upsampling!
hz = linspace(0,EEG.srate,nfft);

% spectral boundary indices
theta = hz>3 & hz<8;
alpha = hz>8 & hz<13;

%% extract power

% obtain Fourier coefficients and extract power spectrum
dataXpre = abs( fft(EEG.data(:,tidxPre,:),nfft,2) ).^2;
dataXpst = abs( fft(EEG.data(:,tidxPst,:),nfft,2) ).^2;

% band-limited power
thetaPre = mean( mean(dataXpre(:,theta,:),2) ,3);
thetaPst = mean( mean(dataXpst(:,theta,:),2) ,3);

alphaPre = mean( mean(dataXpre(:,alpha,:),2) ,3);
alphaPst = mean( mean(dataXpst(:,alpha,:),2) ,3);

% compute ratios
ratPre = alphaPre./thetaPre;
ratPst = alphaPst./thetaPst;

%% now for plotting

figure(1), clf

% define color limits for topoplots
rawclim = [-1 1]*.75 + 1;
logclim = [-1 1]*.5;

%%% first in raw units
subplot(231)
topoplotIndie(ratPre,EEG.chanlocs,'electrodes','off','numcontour',0);
title('\alpha/\theta raw PRE')
set(gca,'clim',rawclim)

subplot(232)
topoplotIndie(ratPst,EEG.chanlocs,'electrodes','off','numcontour',0);
title('\alpha/\theta raw POST')
set(gca,'clim',rawclim)

subplot(233)
topoplotIndie(ratPst-ratPre,EEG.chanlocs,'electrodes','off','numcontour',0);
title('\alpha/\theta raw Post-Pre')
set(gca,'clim',[-1 1]*.2)


%%% repeat for log scaled
subplot(234)
topoplotIndie(log10(ratPre),EEG.chanlocs,'electrodes','off','numcontour',0);
title('log_{10}(\alpha/\theta) PRE')
set(gca,'clim',logclim)

subplot(235)
topoplotIndie(log10(ratPst),EEG.chanlocs,'electrodes','off','numcontour',0);
title('log_{10}(\alpha/\theta) POST')
set(gca,'clim',logclim)

subplot(236)
topoplotIndie(log10(ratPst)-log10(ratPre),EEG.chanlocs,'electrodes','off','numcontour',0);
title('log_{10}(\alpha/\theta) Post-Pre')
set(gca,'clim',logclim/5)

%% done.
