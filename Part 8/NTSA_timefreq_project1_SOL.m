%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-frequency analyses
%      VIDEO: Project 4-1: Solutions
% Instructor: sincxpress.com
%
%%

%% 

load v1_laminar.mat

chanidx = 7;

% wavelet parameters
minFreq =  3;
maxFreq = 80;
nFrex   = 40;
frex    = logspace(log10(minFreq),log10(maxFreq),nFrex);

% other wavelet parameters
wtime = -1:1/srate:1;
halfwav = floor(length(wtime)/2)-1;


% baseline time window
baseline_time = [ -.4 -.1 ];

% FFT parameters
n_wave = length(wtime);
n_data = size(csd,2)*size(csd,3);
n_conv = n_wave + n_data - 1;

%% create non-phase-locked dataset

% compute ERP
erp = mean(csd(chanidx,:,:),3);

% compute induced power by subtracting ERP from each trial
nonphaselocked = squeeze(csd(chanidx,:,:)) - repmat(erp',1,size(csd,3));

% FFT of data
dataX{1} = fft(reshape(csd(chanidx,:,:),1,[]),n_conv); % total 
dataX{2} = fft(reshape(nonphaselocked,1,[]),n_conv); % induced

% convert baseline from ms to indices
bidx = dsearchn(timevec',baseline_time');

%% run convolution

% initialize output time-frequency data
tf = zeros(2,length(frex),size(csd,2));

for fi=1:length(frex)
    
    % create wavelet and take its FFT
    cmw  = exp(2*1i*pi*frex(fi).*wtime) .* exp(-4*log(2)*wtime.^2./.2^2);
    cmwX = fft(cmw,n_conv);
    
    % run convolution for each of total, non-phase-locked, and phase-locked
    for i=1:2
        
        % convolution...
        convres = ifft(cmwX.*dataX{i},n_conv);
        convres = convres(halfwav+1:end-halfwav-1);
        
        % reshape back to timeXtrials
        convres = reshape(convres,size(csd,2),size(csd,3));
        
        % compute power
        tf(i,fi,:) = mean(abs(convres).^2,2);
        
        % db correct power
        tf(i,fi,:) = 10*log10( squeeze(tf(i,fi,:)) ./ mean(tf(i,fi,bidx(1):bidx(2)),3) );
        
    end % end loop around total/nonphase-locked/phaselocked
end % end frequency loop

%% plotting

analysis_labels = {'Total';'Non-phase-locked'};

% color limits
clim = [ -15 15 ];

% scale ERP for plotting
erpt = (erp-min(erp))./max(erp-min(erp));
erpt = erpt*(frex(end)-frex(1))+frex(1);
erpt = erpt/2+10;

figure(1), clf
for i=1:2
    
    subplot(1,3,i), hold on
    contourf(timevec,frex,squeeze(tf(i,:,:)),40,'linecolor','none')
    
    set(gca,'clim',clim,'xlim',[-.2 1.5])
    xlabel('Time (ms)')
    ylabel('Frequency (Hz)')
    title(analysis_labels{i})
    axis square
    colorbar

    % plot ERP on top
    if i==1
        plot(timevec,erpt,'k')
    end
end

subplot(133), hold on
contourf(timevec,frex,squeeze(tf(1,:,:)-tf(2,:,:)),40,'linecolor','none')
set(gca,'clim',clim/10,'xlim',[-.2 1.5])
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Phase-locked')
colorbar
axis square

% plot ERP on top
plot(timevec,erpt,'k')

%% end.
