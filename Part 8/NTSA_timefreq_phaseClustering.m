%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Inter-trial phase clustering before vs. after removing ERP
% Instructor: sincxpress.com
%
%%

% load in data
load sampleEEGdata.mat


% subtract the ERP from the single trials
EEG.subdata = zeros(size(EEG.data));

for chani=1:EEG.nbchan
    
    % ERP from this channel
    erp = mean(EEG.data(chani,:,:),3);
    
    for triali=1:EEG.trials
        EEG.subdata(chani,:,triali) = EEG.data(chani,:,triali) - erp;
    end % end trial loop
end % end channel loop


% do it all in one line ;)
EEG.subdata2 = bsxfun(@minus,EEG.data,mean(EEG.data,3));


%% setup wavelet parameters

% frequency parameters
min_freq =  2;
max_freq = 30;
num_frex = 40;
frex = linspace(min_freq,max_freq,num_frex);

% which channel to plot
channel2use = 'o1';

% other wavelet parameters
range_cycles = [ 4 10 ];

s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);
wavtime = -2:1/EEG.srate:2;
half_wave = (length(wavtime)-1)/2;


% FFT parameters
nWave = length(wavtime);
nData = EEG.pnts * EEG.trials;
nConv = nWave + nData - 1;



% now compute the FFT of all trials concatenated
alldata  = reshape( EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:) ,1,[]);
dataX    = fft( alldata ,nConv );
dataXsub = fft( reshape( EEG.subdata(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:) ,1,[]) ,nConv );


% initialize output time-frequency data
tf = zeros(2,num_frex,EEG.pnts);

%% now perform convolution

% loop over frequencies
for fi=1:length(frex)
    
    %% create wavelet and get its FFT
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    %% convolution for full data
    as = ifft(waveletX .* dataX);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, EEG.pnts, EEG.trials );
    
    % compute ITPC
    tf(1,fi,:) = abs(mean(exp(1i*angle(as)),2));
    
    %% repeat for reduced (residual) data
    as = ifft(waveletX .* dataXsub);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, EEG.pnts, EEG.trials );
    
    % compute ITPC
    tf(2,fi,:) = abs(mean(exp(1i*angle(as)),2));
    
end

%% plot the results

ERPlabel = {'IN';'EX'};

figure(1), clf
for i=1:2
    subplot(2,1,i)
    contourf(EEG.times,frex,squeeze(tf(i,:,:)),40,'linecolor','none')
    set(gca,'clim',[0 .7],'xlim',[-500 1300])
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
    title([ 'ITPC with ERP ' ERPlabel{i} 'cluded' ])
end

colormap hot

%% done.
