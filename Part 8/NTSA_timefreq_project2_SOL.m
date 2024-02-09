%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Project 4-2: Solutions
% Instructor: sincxpress.com
%
%%

load sampleEEGdata.mat

nyquist = EEG.srate/2;

%% TF analysis

nFrex  = 40;
frex   = linspace(5,40,nFrex);
fwidth = linspace(2,8,nFrex);

% baseline
bidx = dsearchn(EEG.times',[-500 -200]');


% strung-out data
longdata = double( reshape(EEG.data(30,:,:),1,[]) );

% initialize output matrices
tf = zeros(nFrex,EEG.pnts);
filtpow = zeros(nFrex,2000);

for fi=1:nFrex
    
    %% create and characterize filter
    
    % create the filter
    frange = [frex(fi)-fwidth(fi) frex(fi)+fwidth(fi)];
    order  = round( 20*EEG.srate/frange(1) );
    fkern  = fir1(order,frange/nyquist);
    
    % compute the power spectrum of the filter kernel
    filtpow(fi,:) = abs(fft(fkern,2000)).^2;
    
    %% apply to data
    
    
    % apply to data
    filtdat = filtfilt(fkern,1,longdata);
    
    % hilbert transform
    hdat = hilbert(filtdat);
    
    % extract power
    powdat = abs( reshape(hdat,EEG.pnts,EEG.trials) ).^2;
    
    % trial average and put into TF matrix
    base = mean(mean(powdat(bidx(1):bidx(2),:),2),1);
    tf(fi,:) = 10*log10( mean(powdat,2)/base );
    
end

%% visualize the filter gains

figure(1), clf
imagesc([0 EEG.srate],frex([1 end]),filtpow)
set(gca,'ydir','normal','xlim',[0 frex(end)+10])
xlabel('Frequency (Hz)'), ylabel('filter center frequency (Hz)')

%% visualize TF matrix

figure(2), clf
contourf(EEG.times,frex,tf,40,'linecolor','none')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
set(gca,'xlim',[-200 1300],'clim',[-3 3])
colorbar

%% done.
