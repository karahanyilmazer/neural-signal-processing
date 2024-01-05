%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Program the inverse Fourier transform from scratch!
% Instructor: sincxpress.com
%
%%

% random signal
N       = 50;          % length of sequence
signal  = sin(linspace(0,4*pi,N)) + randn(1,N)/2;  % data
fTime   = ((1:N)-1)/N; % "time" used in Fourier transform

% initialize Fourier output matrix
fourierCoefs = zeros(size(signal)); 

% loop over frequencies
for fi=1:N
    
    % complex sine wave for this frequency
    fourierSine = exp(-1i*2*pi*(fi-1)*fTime);
    
    % dot product as sum of point-wise multiplications
    fourierCoefs(fi) = sum( fourierSine.*signal );
end

% divide by N to scale coefficients properly
fourierCoefs = fourierCoefs / N;

% use the fft function on the same data for comparison
fourierCoefsF = fft(signal) / N;


%% plotting

figure(1), clf

% time domain signal
subplot(211)
plot(signal)
xlabel('Time (a.u.)')
title('Data')

% amplitude of "manual" Fourier transform
subplot(212), hold on
plot(abs(fourierCoefs)*2,'*-')

% amplitude of FFT
plot(abs(fourierCoefsF)*2,'ro')


xlabel('Frequency (a.u.)')
legend({'Manual FT';'FFT'})

%% now for the IFFT

reconSig = zeros(size(signal));

for fi=1:N
    
    % create "template" sine wave
    fourierSine = 
    
    % modulate by fourier coefficient and add to reconstructed signal
    reconSig = 
end
    
figure(2), clf, hold on

plot(signal,'b*-')
plot(real(reconSig),'ro')
plot(real(ifft(fourierCoefsF))*N,'ko')
xlabel('Time (a.u.)')

%% done.
