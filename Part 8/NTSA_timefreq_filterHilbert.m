%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Narrowband filtering and the Hilbert transform
% Instructor: sincxpress.com
%
%%


%% design FIR filter via fir1

srate   = 1000;
nyquist = srate/2;
frange  = [10 15];
order   = round( 20*srate/frange(1) );

% filter kernel
filtkern = fir1(order,frange/nyquist);

% compute the power spectrum of the filter kernel
filtpow = abs(fft(filtkern)).^2;
% compute the frequencies vector and remove negative frequencies
hz      = linspace(0,srate/2,floor(length(filtkern)/2)+1);
filtpow = filtpow(1:length(hz));



figure(1), clf
subplot(131)
plot(filtkern,'linew',2)
xlabel('Time points')
title('Filter kernel (fir1)')
axis square


% plot amplitude spectrum of the filter kernel
subplot(132), hold on
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


subplot(133), hold on
plot(hz,10*log10(filtpow),'ks-','linew',2,'markersize',10,'markerfacecolor','w')
plot([1 1]*frange(1),get(gca,'ylim'),'k:')
set(gca,'xlim',[0 frange(1)*4],'ylim',[-80 2])
xlabel('Frequency (Hz)'), ylabel('Filter gain (dB)')
title('Frequency response of filter (fir1)')
axis square

%% apply the filter to the data

pnts = 10000;
origdata = randn(pnts,1);
timevec = (0:pnts-1)/srate;

filtdat = filtfilt(filtkern,1,origdata);

%% Hilbert transform

hildat = hilbert(filtdat);

figure(2), clf
subplot(311), hold on
plot(timevec,filtdat,'k')
plot(timevec,real(hildat),'ro')
zoom on
title('Filtered signal')
legend({'Original';'Real'})

subplot(312)
plot(timevec,real(hildat), timevec,imag(hildat))
title('Complex representation')
legend({'Real';'Imag'})

subplot(313)
plot(timevec,abs(hildat))
xlabel('Time (s)'), ylabel('Amplitude')
title('Band-limited amplitude envelope')

%% done.
