%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Project 4-2: Time-frequency power plot via filter-Hilbert
% Instructor: sincxpress.com
%
%%

load sampleEEGdata.mat

nyquist = EEG.srate/2;

%% TF analysis

% frequency parameters
nFrex  = 40;
frex   = linspace(5,40,nFrex);
fwidth = linspace(2,8,nFrex);

% baseline
bidx = 


% strung-out data
longdata = 

% initialize output matrices
tf = 
filtpow = 

for fi=1:nFrex
    
    %% create and characterize filter
    
    % create the filter
    
    
    % compute the power spectrum of the filter kernel
    
    
    %% apply to data
    
    
    % apply to data
    
    % hilbert transform
    
    
    % extract power
    
    
    % trial average and put into TF matrix
    
    
end

%% visualize the filter gains

figure(1), clf
imagesc(
set(gca,'ydir','normal','xlim',[0 frex(end)+10])
xlabel('Frequency (Hz)'), ylabel('filter center frequency (Hz)')

%% visualize TF matrix

figure(2), clf
contourf(
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
set(gca,'xlim',[-200 1300],'clim',[-3 3])
colorbar

%% done.
