%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Comparing average spectra vs. spectra of an average
% Instructor: sincxpress.com
%
%%

% Compute the power spectrum of data from one electrode.
%  First, compute the power spectrum separately for each trial and then average the power 
%    results together. 
%  Next, average the trials together and then compute the power spectrum. 
%  Be able to restrict the time window and spectral resolution.

% load the LFP data
load v1_laminar.mat

% pick which channel
chan2use = 7;

figure(1), clf, hold on
h = plot(timevec,squeeze(csd(chan2use,:,:)));
plot(timevec,mean(csd(chan2use,:,:),3),'k','linew',2);
set(gca,'xlim',[-.5 1.3],'ylim',[-2500 3500])
set(h,'Color',ones(1,3)*.7)
xlabel('Time (s)'), ylabel('Activity (\muV)')

%% FFTs

% time window for FFT in seconds and indices
timewin = [ 0 .5 ]; % in seconds
tidx = dsearchn(timevec',timewin'); % converted to indices

% specify spectral resolution
spectres = .5; % Hz
nfft = round( srate/spectres ); 



% FFT of all trials individually (note that we can do it in one line)
powspectSeparate = fft(squeeze(csd(chan2use,tidx(1):tidx(2),:)),nfft)/length(timevec);
powspectSeparate = mean(2*abs(powspectSeparate),2); % average over trials, not over frequency!

% now FFT of all trials after averaging together
powspectAverage  = fft(squeeze(mean(csd(chan2use,tidx(1):tidx(2),:),3)),nfft)/length(timevec);
powspectAverage  = 2*abs(powspectAverage);

% frequencies in Hz
hz = linspace(0,srate/2,floor(nfft/2)+1);


% now plot
figure(2), clf, hold on
plot(hz,powspectSeparate(1:length(hz)),'linew',2)
plot(hz,powspectAverage(1:length(hz)),'linew',2)

set(gca,'xlim',[0 100],'ylim',[0 100])
xlabel('Frequency (Hz)'), ylabel('Amplitude (\muV)')
legend({'mean(abs(fft))';'abs(mean(fft))'})

title([ 'Power spectra between ' num2str(timewin(1)) '-' num2str(timewin(2)) 's' ])

%% done.
