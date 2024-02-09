%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Instantaneous frequency in simulated and in real data
% Instructor: sincxpress.com
%
%%

% see also: http://mikexcohen.com/data/Cohen2014_freqslide.pdf

% simulation details
srate = 1000;
time  = (0:4*srate-1)/srate;
pnts  = length(time);
noise = 0; % noise amplitude


% create signal (multipolar chirp)
k = 10; % poles for frequencies
freqmod = 20*interp1(rand(1,k),linspace(1,k,pnts),'pchip');
signal  = sin( 2*pi * ((time + cumsum(freqmod))/srate) );
signal  = signal + randn(size(signal))*noise;


% compute instantaneous frequency
angels = angle(hilbert(signal));
instfreq = diff(unwrap(angels))/(2*pi/srate);


% static power spectrum
hz  = linspace(0,srate/2,floor(pnts/2)+1);
amp = 2*abs(fft(signal)/pnts);
amp = amp(1:length(hz)).^2;


%% plotting

figure(1), clf

% signal
subplot(311)
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain')

% static power spectrum
subplot(312)
plot(hz,amp,'ks-','linew',2,'markerfacecolor','w')
title('Frequency domain')
set(gca,'xlim',[0 max(freqmod)*1.2])
xlabel('Frequency (Hz)'), ylabel('Amplitude')

% IF
subplot(313), hold on
plot(time(1:end-1),instfreq,'k','linew',2)
plot(time,freqmod,'r','linew',2)
set(gca,'ylim',[0 max(freqmod)*1.2])
xlabel('Time (s)'), ylabel('Frequency (Hz)')
legend({'Estimated';'Ground truth'})
title('Instantaneous frequency')

%% clean the IF: mean-smoothing

% mean smoothing kernel
k = 13;
kernel = ones(1,k)/k;
nfft = pnts + k - 1;
instfreqSmooth = ifft( fft(kernel,nfft).*fft(instfreq,nfft) );
instfreqSmooth = instfreqSmooth(ceil(k/2):end-ceil(k/2));

plot(time(1:end-1),instfreqSmooth,'b','linew',3)

%% in real data!

load EEGrestingState.mat
time = (0:length(eegdata)-1)/srate;

figure(2), clf
pwelch(eegdata,srate,srate/2,srate*2,srate)
set(gca,'xlim',[0 40])


%% narrowband filter around 10 Hz (FIR)

% filter parameters
nyquist = srate/2;
frange  = [8 12];
order   = round( 20*srate/frange(1) );

% filter kernel
filtkern = fir1(order,frange/nyquist);

% compute the power spectrum of the filter kernel
filtpow = abs(fft(filtkern)).^2;
% compute the frequencies vector and remove negative frequencies
hz      = linspace(0,srate/2,floor(length(filtkern)/2)+1);
filtpow = filtpow(1:length(hz));


%%% visualize the filter kernel
figure(3), clf
subplot(121)
plot(filtkern,'linew',2)
xlabel('Time points')
title('Filter kernel (fir1)')
axis square

% plot amplitude spectrum of the filter kernel
subplot(122), hold on
plot(hz,filtpow,'ks-','linew',2,'markerfacecolor','w')
plot([0 frange(1) frange frange(2) nyquist],[0 0 1 1 0 0],'ro-','linew',2,'markerfacecolor','w')

% dotted line corresponding to the lower edge of the filter cut-off
plot([1 1]*frange(1),get(gca,'ylim'),'k:')

% make the plot look nicer
set(gca,'xlim',[0 frange(1)*4])%,'ylim',[-.05 1.05])
xlabel('Frequency (Hz)'), ylabel('Filter gain')
legend({'Actual';'Ideal'})
title('Frequency response of filter (fir1)')
axis square


%% IF on filtered data

% apply the filter to the data
feegdata = filtfilt(filtkern,1,double(eegdata));

% plot the data for comparison
figure(4), clf
plot(time,eegdata, time,feegdata ,'linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
zoom on



% compute instantaneous frequency
angels = angle(hilbert(feegdata));
instalpha = diff(unwrap(angels))/(2*pi/srate);

figure(5), clf
plot(time(1:end-1),instalpha,'ks-','markerfacecolor','w')
xlabel('Time (s)'), ylabel('Frequency (Hz)')
zoom on

%% apply median filter to supra-threshold points

% convert to z and show histogram
instz = (instalpha-mean(instalpha)) / std(instalpha);

figure(6), clf
hist(instz,200)
xlabel('I.F. (z)'), ylabel('Count')
title('Distribution of z-norm. IF')

% identify supra-threshold data points
tofilter = find( abs(instz)>2 );

% now for median filter
instalphaFilt = instalpha;
k = round(1000*50/srate); % median kernel size is 2k+1, where k is time in ms
for i=1:length(tofilter)
    indices = max(1,tofilter(i)-k):min(pnts,tofilter(i)+k);
    instalphaFilt(tofilter(i)) = median(instalpha(indices));
end

figure(7), clf, hold on
plot(time(1:end-1),instalpha,'k','linew',2)
plot(time(1:end-1),instalphaFilt,'r--','linew',2)
xlabel('Time (s)'), ylabel('Frequency (Hz)')
legend({'Original';'Filtered'})
set(gca,'ylim',[5 15])
zoom on

%% done.
