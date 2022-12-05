%%
%   COURSE: Neural signal processing and analysis: Zero to hero
%  SESSION: Introduction
%  TEACHER: Mike X Cohen, sincxpress.com
%

%% become familiar with the sample data used in class
%

%% start with the EEG data
load('sampleEEGdata.mat')

% explore a bit... what are the different fields? What is the size of the data?
% How many channels/time points/trials?
% What is the earliest and last time point? 
% Where is time = 0?
% What is the sampling rate?
EEG

%% plot ERPs and topographical maps

% compute the ERP of each channel 
% (remember that the ERP is the time-domain average across all trials at each time point)
erp = mean(EEG.data, 3);
    

% pick a channel and plot ERP
chan2plot = 'fcz';

figure(1), clf
plot(EEG.times,erp( strcmpi({EEG.chanlocs.labels},chan2plot) ,:),'linew',2)
xlabel('Time (ms)'), ylabel('Activity (\muV)')
set(gca,'xlim',[-400 1200])

%% plot topographical maps

time2plot = 300; % in ms

% convert time in ms to time in indices
[~,tidx] = min(abs(EEG.times-time2plot));

% make a topographical map of the ERP at only this time point.
figure(2), clf
topoplotIndie(erp,EEG.chanlocs);
title([ 'ERP from ' num2str(time2plot) ' ms' ])
colorbar

%%

%% now for sample CSD V1 data

load v1_laminar

% check out the variables in this mat file, using the function whos
% If you don't know what variables are in this file vs. already in the workspace,
%  you can clear the workspace and then load the file in again.

% the data are in variable csd. What is the size? How many channels/time points/trials?

%%

% % plot ERP from channel 7 in one line of code!
% figure(3), clf
% plot(timevec,
% hold on
% plot(get(gca,'xlim'),[0 0],'k--')
% plot([0 0],get(gca,'ylim'),'k--')
% plot([0 0]+.5,get(gca,'ylim'),'k--')
% xlabel('Time (s)'), ylabel('Activity (\muV)')
% set(gca,'xlim',[-.1 1.4])
% 
% 
% % plot depth-by-time image of ERP by averaging over trials
% figure(4), clf
% contourf(timevec,1:16,squeeze(mean(csd,3)),40,'linecolor','none')
% set(gca,'xlim',[0 1.3])
% xlabel('Time (sec.)'), ylabel('Cortical depth')

%% done.
