%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Generating transient oscillations
% Instructor: sincxpress.com
%
%%

%% simulation details

pnts  = 4000;
srate = 1000;
time  = (0:pnts-1)/srate - 1;

% gaussian parameters
peaktime = 1; % seconds
fwhm     = .4;

% sine wave parameters
sinefreq = 7; % for sine wave

%% create signal

% create Gaussian taper
gaus = exp( -(4*log(2)*(time-peaktime).^2) / fwhm^2 );

% sine wave with random phase value ("non-phase-locked")
cosw = cos(2*pi*sinefreq*time + 2*pi*rand);

% signal
signal = cosw .* gaus;


% and plot
figure(1), clf
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')

%%
