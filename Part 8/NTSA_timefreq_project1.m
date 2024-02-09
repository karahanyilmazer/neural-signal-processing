%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Project 4-1: Phase-locked, non-phase-locked, and total power
% Instructor: sincxpress.com
%
%%

%% 

load v1_laminar.mat

chanidx = 7;

% wavelet parameters
minFreq = 
maxFreq = 
nFrex   = 
frex    = 

% other wavelet parameters
wtime = 
halfwav = 


% baseline time window
baseline_time = [ -.4 -.1 ];

% FFT parameters
n_wave = 
n_data = 
n_conv = 

%% create non-phase-locked dataset

% compute ERP
erp = 

% compute induced power by subtracting ERP from each trial
nonphaselocked = 

% FFT of data
dataX{1} = fft(reshape(csd(
dataX{2} = fft(reshape(nonphaselocked(

% convert baseline from ms to indices
bidx = dsearchn(timevec',baseline_time');

%% run convolution

% initialize output time-frequency data
tf = zeros();

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet
    cmw = 
    % take FFT of data
    
    
    % run convolution for total and non-phase-locked
    for i=1:2
        
        % convolution...
        convres = ifft(
        
        % reshape back to timeXtrials
        convres = 
        
        % compute power
        tf(i,fi,:) = mean(abs(
        
        % db correct power
        tf(i,fi,:) = 10*log10( 
        
    end % end loop around total/nonphase-locked/phaselocked
end % end frequency loop

%% plotting

analysis_labels = {'Total';'Non-phase-locked'};

% color limits
clim = 

% scale ERP for plotting
erpt = (erp-min(erp))./max(erp-min(erp));


figure(1), clf
for i=1:2
    
    subplot(1,3,i), hold on
    contourf(
    
    set(gca,'clim',clim,'xlim',[-.2 1.5])
    xlabel('Time (ms)')
    ylabel('Frequency (Hz)')
    title(analysis_labels{i})
    axis square
    colorbar

    % plot ERP on top
    plot(timevec,erpt,'k')
end

% phase-locked component is the difference between total and non-phase-locked
subplot(133)
contourf
set(gca,'clim',clim/10,'xlim',[-.2 1.5])
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Phase-locked')
colorbar
axis square

%% end.
