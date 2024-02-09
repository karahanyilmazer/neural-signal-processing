%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Baseline normalize power with dB and % change
% Instructor: sincxpress.com
%
%%

%% compare dB and %change in simulated numbers

activity = 1:.01:20; % activity
baseline = 10; % baseline

% create normalized vectors
db = 10*log10( activity ./ baseline );
pc = 100*( activity-baseline)./ baseline;


figure(1), clf, hold on
plot(activity,'linew',3)
plot(db,'r','linew',3)
legend({'"activity"','dB'})


% compare dB and percent change directly
figure(2), clf, hold on
plot(db,pc,'k','linew',2)
xlabel('dB'), ylabel('Percent change')

% find indices where db is closest to -/+2
[~,dbOfplus2]  = min(abs(db-+2));
[~,dbOfminus2] = min(abs(db--2));

% plot as guide lines
axislim=axis;
plot([db(dbOfplus2)  db(dbOfplus2)], [pc(dbOfplus2)  axislim(3)],'k',[axislim(1) db(dbOfplus2)], [pc(dbOfplus2)  pc(dbOfplus2)], 'k')
plot([db(dbOfminus2) db(dbOfminus2)],[pc(dbOfminus2) axislim(3)],'k',[axislim(1) db(dbOfminus2)],[pc(dbOfminus2) pc(dbOfminus2)],'k')

%% now for real data!

load sampleEEGdata.mat

% baseline window and convert into indices
basewin = [-500 -200];
baseidx = dsearchn(EEG.times',basewin');

%% setup wavelet parameters

% frequency parameters
min_freq =  2;
max_freq = 30;
num_frex = 40;
frex = linspace(min_freq,max_freq,num_frex);

% which channel to plot
channel2use = 'o1';

% other wavelet parameters
fwhms = logspace(log10(.6),log10(.3),num_frex);
wavtime = -2:1/EEG.srate:2;
half_wave = (length(wavtime)-1)/2;


% FFT parameters
nWave = length(wavtime);
nData = EEG.pnts * EEG.trials;
nConv = nWave + nData - 1;


% now compute the FFT of all trials concatenated
alldata = reshape( EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:) ,1,[]);
dataX   = fft( alldata ,nConv );


% initialize output time-frequency data
tf = zeros(num_frex,EEG.pnts);

%% now perform convolution

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* ...
               exp(-4*log(2)*wavtime.^2./fwhms(fi)^2);
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    % now run convolution in one step
    as = ifft(waveletX .* dataX);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, EEG.pnts, EEG.trials );
    
    % compute power and average over trials
    tf(fi,:) = mean( abs(as).^2 ,2);
end

%% baseline normalization

% baseline power
baseline = mean( tf(:,baseidx(1):baseidx(2)) ,2);

% decibel
tfdb = 10*log10( bsxfun(@rdivide, tf, baseline) );

% percent change
tfpc = 100 * bsxfun(@rdivide, bsxfun(@minus,tf,baseline), baseline);


%% plot results

% define color limits
climraw = [0 2];
climdb  = [-3 3];
climpct = [-90 90];


figure(3), clf

% raw power
subplot(231)
contourf(EEG.times,frex,tf,40,'linecolor','none')
set(gca,'clim',climraw,'xlim',[-300 1000])
xlabel('Time (ms)'), ylabel('Frequency (Hz)'), axis square
title('Raw power')

% db
subplot(232)
contourf(EEG.times,frex,tfdb,40,'linecolor','none')
set(gca,'clim',climdb,'xlim',[-300 1000])
xlabel('Time (ms)'), ylabel('Frequency (Hz)'), axis square
title('dB power')

% percent change
subplot(233)
contourf(EEG.times,frex,tfpc,40,'linecolor','none')
set(gca,'clim',climpct,'xlim',[-300 1000])
xlabel('Time (ms)'), ylabel('Frequency (Hz)'), axis square
title('Percent change power')

%% empirical comparisons

subplot(223)
plot(tf(:),tfpc(:),'rs','markerfacecolor','k')
xlabel('Raw'), ylabel('Pct change')
axis square
title('Raw power vs. pct change power')


subplot(224)
plot(tfdb(:),tfpc(:),'rs','markerfacecolor','k')
ylabel('%\Delta'), xlabel('DB')
axis square
title('Pct change power vs. dB power')

%% done.
