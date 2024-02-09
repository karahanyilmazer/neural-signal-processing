%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Exploring wavelet parameters in simulated data
% Instructor: sincxpress.com
%
%%

%% simulate data as transient oscillation

pnts  = 4000;
srate = 1000;
stime  = (0:pnts-1)/srate - 2;

% gaussian parameters
fwhm = .4;

% sine wave parameters
sinefreq = 10; % for sine wave

% create signal
gaus   = exp( -(4*log(2)*stime.^2) / fwhm^2 );
cosw   = cos(2*pi*sinefreq*stime + 2*pi*rand);
signal = cosw .* gaus;

% get signal amplitude
sigamp = abs(hilbert(signal));


figure(1), clf, hold on
plot(stime,signal,'k','linew',3)
plot(stime,sigamp,'r--','linew',1)
xlabel('Time (s)')

%% comparing fixed number of wavelet cycles

% wavelet parameters
num_frex = 50;
min_freq =  2;
max_freq = 20;


% set a few different wavelet widths (FWHM parameter)
fwhms = [ .1 .5 2 ];

% other wavelet parameters
frex = linspace(min_freq,max_freq,num_frex);
wtime = -2:1/srate:2;
half_wave = (length(wtime)-1)/2;

% FFT parameters
nKern = length(wtime);
nConv = nKern+pnts-1;

% initialize output time-frequency data
tf = zeros(length(fwhms),length(frex),pnts);


% FFT of data (doesn't change on frequency iteration)
dataX = fft( signal ,nConv);


% loop over cycles
for fwhmi=1:length(fwhms)
    
    for fi=1:length(frex)
        
        % create wavelet and get its FFT
        cmw  = exp(2*1i*pi*frex(fi).*wtime) .* exp(-4*log(2)*wtime.^2./fwhms(fwhmi)^2);
        cmwX = fft(cmw,nConv);
        cmwX = cmwX./max(cmwX);
        
        % run convolution, trim edges, and reshape to 2D (time X trials)
        as = ifft(cmwX.*dataX);
        as = as(half_wave+1:end-half_wave);
        
        % put power data into big matrix
        tf(fwhmi,fi,:) = abs(as);
    end
end


%% plot results

figure(3), clf, colormap parula

% time-frequency plots
for fwhmi=1:length(fwhms)
    subplot(2,3,fwhmi)
    contourf(stime,frex,squeeze(tf(fwhmi,:,:)),40,'linecolor','none')
    title([ 'Wavelet with ' num2str(fwhms(fwhmi)) ' s FWHM' ])
    xlabel('Time (s)'), ylabel('Frequency (Hz)')
end


% show amplitude envelopes at peak frequency
fidx = dsearchn(frex',sinefreq);
subplot(223), hold on
plot(stime,squeeze(tf(:,fidx,:)))
plot(stime,sigamp/2,'k--','linew',2)
xlabel('Time (s)')
legend([regexp(num2str(fwhms),' +','split'),'Truth'])
title('Time domain')


% show spectra at one time point
tidx = dsearchn(stime',0);
subplot(224), hold on
plot(frex,squeeze(tf(:,:,tidx)))
plot(linspace(0,srate,pnts),2*abs(fft(signal)/pnts),'k--','linew',2)
set(gca,'xlim',frex([1 end]))
xlabel('Frequency (Hz)')
legend([regexp(num2str(fwhms),' +','split'),'Truth'])
title('Frequency domain')

%% done.
