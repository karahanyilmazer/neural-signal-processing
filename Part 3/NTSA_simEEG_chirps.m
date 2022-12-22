%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Generating "chirps" (frequency-modulated signals)
% Instructor: sincxpress.com
%
%%

%% simulation details

pnts  = 10000;
srate = 1024;
time  = (0:pnts-1)/srate;

%% chirps

% "bipolar" chirp
freqmod = linspace(5,15,pnts);

% multipolar chirp
% k = 10; % poles for frequencies
% freqmod = 20*interp1(rand(1,k),linspace(1,k,pnts));


% signal time series
signal  = sin( 2*pi * ((time + cumsum(freqmod))/srate) );
% Note: the code in the video has a bug in the previous line,
%   due to incorrect parentheses: srate should scale both time
%   and freqmod, as above.

%% plotting

figure(1), clf

subplot(211)
plot(time,freqmod,'r','linew',3)
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Instantaneous frequency')

subplot(212)
plot(time,signal,'k')
xlabel('Time (s)')
ylabel('Amplitude (a.u.)')
title('Signal (chirp)')

%%
