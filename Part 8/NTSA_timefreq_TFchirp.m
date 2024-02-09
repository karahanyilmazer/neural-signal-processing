%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Create a time-frequency plot of a nonlinear chirp
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

% laplace distribution, scaled to [0 1]
fm = exp( -abs(timev) );
fm = fm-min(fm);
fm = fm./max(fm);

% finally, scale to frequency ranges
fm = fm*(maxfreq-minfreq) + minfreq;

% now for chirp
churp = sin(2*pi*((timev+cumsum(fm))/srate));

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
xlabel('Time (s)'), ylabel('Amplitude (a.u.)')
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
fwhms = logspace(log10(.5),log10(.1),numfrex);

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

%% done.
