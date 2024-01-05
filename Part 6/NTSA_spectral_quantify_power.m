%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Extracting average power from a frequency band
% Instructor: sincxpress.com
%
%%

% load data
load EEGrestingState.mat
N = length(eegdata);


% create Hann window
winsize = 2*srate; % 2-second window
hannw = .5 - cos(2*pi*linspace(0,1,winsize))./2;

% number of FFT points (frequency resolution)
nfft = srate*100;

% call pwelch with outputs
[powspect,hz] = pwelch(eegdata,hannw,round(winsize/4),nfft,srate);

% then you can plot those...
figure(1), clf, hold on
plot(hz,powspect,'k','linew',2)
set(gca,'xlim',[0 80])
xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')

%% extract band-limited power

% frequency boundaries in Hz
freqlims = [ 8 12 ];

% convert to indices
[~,fidx(1)] = min(abs( hz-freqlims(1) ));
[~,fidx(2)] = min(abs( hz-freqlims(2) ));

% extract average power in that range
avepow = mean(powspect(fidx(1):fidx(2)));


% shade region in plot
ph = patch([hz(fidx(1):fidx(2)); hz(fidx(2)-1:-1:fidx(1))],[powspect(fidx(1):fidx(2)); zeros(diff(fidx),1)],'k');
set(ph,'edgecolor','none','FaceAlpha',.3)


%% done.
