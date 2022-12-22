%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Non-stationary narrowband activity via filtered noise
% Instructor: sincxpress.com
%
%%

% simulation details
pnts  = 4567;
srate =  987;

% signal parameters in Hz
peakfreq = 14;
fwhm     =  5;


% frequencies
hz = linspace(0,srate,pnts);


%%% create frequency-domain Gaussian
s  = fwhm*(2*pi-1)/(4*pi); % normalized width
x  = hz-peakfreq;          % shifted frequencies
fg = exp(-.5*(x/s).^2);    % gaussian


% Fourier coefficients of random spectrum
fc = rand(1,pnts) .* exp(1i*2*pi*rand(1,pnts));

% taper with Gaussian
fc = fc .* fg;

% go back to time domain to get EEG data
signal = 2*real( ifft(fc) );


%%% plotting

figure(1), clf
subplot(211)
plot(hz,abs(fc),'k')
set(gca,'xlim',[0 peakfreq*3])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')
title('Frequency domain')


subplot(212)
plot((0:pnts-1)/srate,signal,'b')
title('Time domain')
xlabel('Time (s)'), ylabel('Amplitude')

%%
