%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Visualize time-frequency power from all channels
% Instructor: sincxpress.com
%
%%

% load in data
load sampleEEGdata.mat

% post-analysis temporal downsampling
times2save = -200:25:800; % in ms
tidx = dsearchn(EEG.times',times2save');

% baseline window and convert into indices
basewin = [-500 -200];
baseidx = dsearchn(EEG.times',basewin');

%% setup for TF decomposition

% spectral parameters
minFreq =  3; % hz
maxFreq = 40; % hz
numFrex = 30;

frex = linspace(minFreq,maxFreq,numFrex);

% wavelet parameters
fwhms = logspace(log10(.6),log10(.3),numFrex);
wtime = -2:1/EEG.srate:2;
halfw = (length(wtime)-1)/2;

% FFT parameters
nWave = length(wtime);
nData = EEG.pnts * EEG.trials;
nConv = nWave + nData - 1;

%% create wavelet spectra

% initialize 
cmwX = zeros(numFrex,nConv);

for fi=1:numFrex
    
    % time domain wavelet
    cmw = exp(1i*2*pi*frex(fi)*wtime) .* exp(-4*log(2)*wtime.^2/fwhms(fi).^2);
    
    % its FFT and save the normalized version
    tmp = fft(cmw,nConv);
    cmwX(fi,:) = tmp./max(tmp);
end


%% time-frequency decomposition

% initialize matrix
tf = zeros(EEG.nbchan,numFrex,length(tidx));

% loop over channels
for chani=1:EEG.nbchan
    
    % data from this channel
    chandat = reshape(EEG.data(chani,:,:),1,[]);
    dataX = fft(chandat,nConv);
    
    % loop over frequencies
    for fi=1:numFrex
        
        % convolution
        convres = ifft( dataX.*cmwX(fi,:) );
        
        % trim and reshape to time X trials
        convres = convres(halfw+1:end-halfw);
        convres = reshape(convres,EEG.pnts,EEG.trials);
        
        % trial-average power
        avepow = mean( abs(convres).^2 ,2);
        
        % baseline power
        base = mean(avepow(baseidx(1):baseidx(2)));
        
        % enter into matrix as dB
        tf(chani,fi,:) = 10*log10( avepow(tidx)/base );
    
    end % end frequency loop
end % end channel loop

%% view one channel and topoplot at a time

% human-readable terms
chan2plot = 'pz'; % channel label
time2plot = 200; % in ms
freq2plot =  12; % in hz

% indices
chanidx = strcmpi(chan2plot,{EEG.chanlocs.labels});
timeidx = dsearchn(times2save',time2plot);
freqidx = dsearchn(frex',freq2plot);

% now plot
figure(1), clf
subplot(121)
contourf(times2save,frex,squeeze(tf(chanidx,:,:)),40,'linecolor','none')
set(gca,'clim',[-1 1]*3), axis square
title([ 'TF power from channel ' chan2plot ])

subplot(122)
topoplotIndie(squeeze(tf(:,freqidx,timeidx)),EEG.chanlocs);
title([ 'Topo at ' num2str(time2plot) ' ms, ' num2str(freq2plot) ' Hz' ])

%% tfviewerx

tfviewerx(times2save,frex,tf,EEG.chanlocs);

%% done.
