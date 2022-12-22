%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: The eeglab EEG structure
% Instructor: sincxpress.com
%
%%

%% familiarize yourself with the sample data and the eeglab structure

% EEG sample data
load sampleEEGdata

% explore a bit..
EEG

%% plot ERPs

% compute the ERP on each channel
erp = mean(EEG.data,3);


% pick a channel and plot ERP
chan2plot = 'fcz';

figure(1), clf
plot(EEG.times,erp( strcmpi({EEG.chanlocs.labels},chan2plot) ,:),'linew',2)

%% plot topographical maps

time2plot = 300; % in ms

% convert time in ms to time in indices
[~,tidx] = min(abs(EEG.times-time2plot));

figure(2), clf
topoplotIndie(erp(:,tidx),EEG.chanlocs);

%% 
