%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Wavelet convolution of close frequencies
% Instructor: sincxpress.com
%
%%

% simulation parameters
srate = 1000;
npnts = srate*3; % 3 seconds
timev = (0:npnts-1)/srate - 1;

% sine wave parameters (in Hz)
freq1 = 10;
freq2 = 13;

% Gaussian parameter (in seconds)
fwhm = .3;
ptime1 = 1;
ptime2 = 1.1;


% generate signals
sig1 = sin(2*pi*freq1*timev) .* exp( -4*log(2)*( (timev-ptime1)/fwhm ).^2 );
sig2 = sin(2*pi*freq2*timev) .* exp( -4*log(2)*( (timev-ptime2)/fwhm ).^2 );

% add a touch of noise
sig1 = sig1 + .1*randn(size(sig1));
sig2 = sig2 + .1*randn(size(sig2));

% now for the signal itself
signal = sig1+sig2;


% plot the signal components
figure(1), clf
subplot(211), hold on
plot(timev,sig1,'b','linew',2)
plot(timev,sig2,'r','linew',2)
xlabel('Time (s)'), ylabel('Amp. (a.u.)')
legend({'sig1';'sig2'})

% and the final signal
subplot(212)
plot(timev,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amp. (a.u.)')


%% time-frequency paramters

minfreq =  2;
maxfreq = 20;
numfrex = 60;

wavtime = -2:1/srate:2;


% vector of wavelet frequencies
frex = logspace(log10(minfreq),log10(maxfreq),numfrex);

% Gaussian parameter
wavefwhm = 2;


nConv = npnts + length(wavtime) - 1;
halfw = floor(length(wavtime)/2)+1;


%% convolution

% initialize
tf = zeros(numfrex,npnts);

% FFT of data (doesn't change over frequencies!)
signalX = fft(signal,nConv);

for fi=1:numfrex
    
    % create complex Morlet wavelet parts
    csw  = exp( 1i* 2*pi*frex(fi)*wavtime );
    gaus = exp( -4*log(2)*wavtime.^2 / wavefwhm^2 );
    
    % wavelet spectrum
    waveX = fft(csw.*gaus,nConv);
    waveX = waveX./max(waveX);
    
    % convolution
    convres = ifft( signalX.*waveX );
    convres = convres(halfw:end-halfw+1);
    
    % extract power
    tf(fi,:) = abs(convres).^2;
end

figure(2), clf
contourf(timev,frex,tf,40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequency (Hz)')
colormap hot

%% done.
