%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Simulating noise
% Instructor: sincxpress.com
%
%%

%% Normally distributed noise

% simulation details
srate  = 100; % sampling rate in Hz
time   = -1:1/srate:2;
pnts   = ;

% frequencies for the power spectrum
hz = linspace(0,srate/2,floor(length(time)/2)+1);


% noise parameters
stretch = 3;
shift   = 0;

% (optional: fix the random-number generator state)
% rng(3);

% generate random data
noise = stretch*randn(size(time)) + shift;

figure(4), clf
subplot(211)
plot(time,noise,'k')
set(gca,'fontsize',15)
title('Normally distributed: Time domain')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(223)
[y,x] = hist(noise,100);
plot(x,y,'k','linew',2)
xlabel('Values'), ylabel('N per bin')
title('Signal histogram (distribution)')
set(gca,'fontsize',15,'xlim',[min(x) max(x)])

subplot(224)
amp = abs(fft(noise)/pnts);
amp(2:end) = 2*amp(2:end);
plot(hz,amp(1:length(hz)),'k')
title('Frequency domain')
set(gca,'fontsize',15)
xlabel('Frequency (Hz)'), ylabel('Amplitude')


%% Pink noise (aka 1/f aka fractal)

%%% questions: what is the effect of changing the "ed" parameter?
%%%            which values make this time-domain signal look most like EEG?

% simulation details for this video
srate = 500; % sampling rate in Hz
time  = -1:1/srate:2;
pnts  = length(time);
hz    = linspace(0,srate/2,floor(length(time)/2)+1);

% generate 1/f amplitude spectrum
ed = 50; % exponential decay parameter
as = rand(1,floor(pnts/2)-1) .* exp(-(1:floor(pnts/2)-1)/ed);
as = [as(1) as 0 0 as(:,end:-1:1)];

% Fourier coefficients
fc = as .* exp(1i*2*pi*rand(size(as)));

% inverse Fourier transform to create the noise
noise = real(ifft(fc)) * pnts;


figure(5), clf
subplot(211)
plot(time,noise,'k')
set(gca,'fontsize',15)
title('Pink noise: Time domain')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(223)
[y,x] = hist(noise,100);
plot(x,y,'k','linew',2)
xlabel('Values'), ylabel('N per bin')
title('Signal histogram (distribution)')
set(gca,'fontsize',15)

subplot(224)
amp = abs(fft(noise)/pnts);
amp(2:end) = 2*amp(2:end);
plot(hz,amp(1:length(hz)),'k')
title('Frequency domain')
set(gca,'fontsize',15)
xlabel('Frequency (Hz)'), ylabel('Amplitude')

%% 
