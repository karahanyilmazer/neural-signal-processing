%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Downsampling time-frequency power
% Instructor: sincxpress.com
%
%%

%% import data

load v1_laminar

% vectors of points to extract from TF result
times2save = -.2:.25:1.2; % in seconds

% convert that to indices
tidx = dsearchn(timevec',times2save');

% baseline time window
bidx = dsearchn(timevec',[-400 -100]');

%% setup wavelet parameters

% frequency parameters
min_freq = 15;
max_freq = 99;
num_frex = 60;
frex = linspace(min_freq,max_freq,num_frex);

% which channel to plot
channel2use = 6;

% other wavelet parameters
fwhms = logspace(log10(.6),log10(.3),num_frex);
wtime = -2:1/srate:2;
halfw = (length(wtime)/2)-1;


% FFT parameters
nWave = length(wtime);
nData = size(csd,2)*size(csd,3);
nConv = nWave + nData - 1;


% now compute the FFT of all trials concatenated
alldata = reshape( csd(channel2use,:,:) ,1,[]);
dataX   = fft( alldata ,nConv );


% initialize output time-frequency data
tffull = zeros(num_frex,length(timevec));
tfdwns = zeros(num_frex,length(tidx));

%% now perform convolution

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    cmw  = exp(2*1i*pi*frex(fi).*wtime) .* ...
           exp(-4*log(2)*wtime.^2./fwhms(fi)^2);
    cmwX = fft(cmw,nConv);
    cmwX = cmwX ./ max(cmwX);
    
    % now run convolution in one step
    as = ifft(cmwX .* dataX);
    as = as(halfw+1:end-halfw-1);
    
    % and reshape back to time X trials
    as = reshape( as, length(timevec), size(csd,3) );
    
    % compute baseline power
    base = mean(mean( abs(as(bidx(1):bidx(2),:)).^2 ,1),2);
    
    % power time series for this frequency
    % baseline-normalized
    powTS = 10*log10( mean(abs(as).^2,2)/base );
    
    % enter full and downsampled data into matrices
    tffull(fi,:) = powTS;
    tfdwns(fi,:) = powTS(tidx);
end

%% plot results

figure(1), clf

clim = [-1 1]*12;

% downsampled TF
subplot(221)
contourf(times2save,frex,tfdwns,40,'linecolor','none')
set(gca,'clim',clim,'xlim',times2save([1 end]))
xlabel('Time (ms)'), ylabel('Frequency (Hz)'), axis square
title('Downsampled TF map')

% full TF
subplot(222)
contourf(timevec,frex,tffull,40,'linecolor','none')
set(gca,'clim',clim,'xlim',times2save([1 end]))
xlabel('Time (ms)'), ylabel('Frequency (Hz)'), axis square
title('Full TF map')

%% comparison for one frequency

% pick a frequency in Hz
freq2plot = 30;

% convert to index
fidx = dsearchn(frex',freq2plot);


subplot(212), cla, hold on
plot(timevec,tffull(fidx,:),'k','linew',2)
plot(times2save,tfdwns(fidx,:),'ro','markerfacecolor','r','markersize',15)
set(gca,'xlim',timevec([1 end]))
legend({'full res.';'downsampled'})
xlabel('Time (s.)'), ylabel('Power (dB)')

%% compare matrix sizes

varinfo = whos('tf*');

figure(2), clf

bar([varinfo.bytes]/1e6)
set(gca,'xlim',[.5 2.5],'XTickLabel',{varinfo.name})
ylabel('megabytes')

%% done.
