%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Compare wavelet-derived spectrum and FFT
% Instructor: sincxpress.com
%
%%

%% create a nonlinear chirp

% simulation parameters
srate = 1001;
pnts  = srate*4;
timev = (0:pnts-1)/srate;
timev = timev - mean(timev);

% chirp parameters (in Hz)
minfreq =  5;
maxfreq = 17;

% exponential function, scaled to [0 1]
fm = exp( timev );
fm = fm-min(fm);
fm = fm./max(fm);

% finally, scale to frequency ranges
fm = fm*(maxfreq-minfreq) + minfreq;

% now for chirp
churp = sin(2*pi*((timev+cumsum(fm))/srate));

% add a pure sine wave
% churp = churp + .4*sin(2*pi*13*timev);

%% plotting

figure(1), clf

% frequency time series
subplot(311)
plot(timev,fm,'k')
xlabel('Time (s)'), ylabel('Frequency (Hz)')

% plot the chirp
subplot(312)
plot(timev,churp,'k')
xlabel('Time (s)'), ylabel('Amplitude (a.u.)')

% power spectrum
subplot(313)
plot(linspace(0,srate,pnts),abs(fft(churp)).^2,'k')
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')
set(gca,'xlim',[0 maxfreq*2])
title('Static FFT')

%% create wavelets

% wavelet parameters
minFreq =  2; % Hz
maxFreq = 30; % Hz
numfrex = 40;

% vector of wavelet frequencies
frex = logspace(log10(minFreq),log10(maxFreq),numfrex);

% Gaussian parameters
fwhms = logspace(log10(.5),log10(.3),numfrex);

% initialize
cmw = zeros(numfrex,pnts);

for fi=1:numfrex

    % create complex sine wave
    csw = exp( 1i* 2*pi*frex(fi)*timev );

    % create the two Gaussian
    gaus = exp( -4*log(2)*timev.^2 / fwhms(fi)^2 );

    % now create the complex Morlet wavelets
    cmw(fi,:) = csw .* gaus;
end

%% convolution

% NOTE: There was a typo in the video; it should be -1 not +1
nConv = 2*pnts - 1;
halfw = floor(pnts/2);

% initialize
tf = zeros(numfrex,pnts);

% FFT of data (doesn't change over frequencies!)
dataX = fft(churp,nConv);

for fi=1:numfrex

    % wavelet spectrum
    waveX = fft(cmw(fi,:),nConv);
    waveX = waveX./max(waveX);

    % convolution
    convres = ifft( dataX.*waveX );
    convres = convres(halfw:end-halfw);

    % extract power
    tf(fi,:) = abs(convres).^2;
end

figure(2), clf
contourf(timev,frex,tf,40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%% comparison

% FFT
fftpower = (2*abs(fft(churp)/pnts)).^2;
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(3), clf, hold on
plot(hz,fftpower(1:length(hz)),'k','linew',2)
plot(frex,mean(tf,2),'r','linew',2)
set(gca,'xlim',[0 maxfreq*2])
xlabel('Frequency (Hz)'), ylabel('Power')
legend({'FFT';'Wavelet'})

%% done.
