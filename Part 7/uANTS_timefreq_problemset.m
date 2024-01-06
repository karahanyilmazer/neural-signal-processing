%%
%   COURSE: Neural signal processing and analysis: Zero to hero
%  SESSION: Time-frequency analysis: Problem set
%  TEACHER: Mike X Cohen, sincxpress.com
%



%% 1) Power and phase from the famous "trial 10"
%    Create a family of complex Morlet wavelets that range from 10 Hz to 100 Hz in 43 linearly spaced steps. 
%    Perform convolution between the wavelets and V1 data from trial 10 for all channels.
%    Extract power and phase, store the results in a channel X frequency X time X pow/phs (thus, 4D) matrix.

clear

load v1_laminar.mat

% soft-coded parameters
freqrange  = [10 100]; % extract only these frequencies (in Hz)
numfrex    = .1;       % number of frequencies between lowest and highest
whichTrial = 10;


% set up convolution parameters
wavtime = -200:1/srate:2-1/srate;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nData   = length(timevec);
nKern   = length(wavtime);
nConv   = nData + nKern - 1;
halfwav = (length(wavtime)-1)/2;

% number of cycles
numcyc = linspace(3,15,numfrex);


% create wavelets
cmwX = zeros(numfrex,nConv);
for fi=1:numfrex
    
    % create time-domain wavelet
    twoSsquared = 2 * () ^ 2;
    cmw = exp(2*1i*pi*frex(fi).*wavtime) .* exp( (-wavtime.^2) / twoSsquared );
    
    % compute fourier coefficients of wavelet and normalize
    cmwX(fi,:) = fft(cmw,nConv);
    cmwX(fi,:) = cmwX(fi,:) ./ max(cmwX(fi,:));
end


% initialize time-frequency output matrix
tf = zeros(size(csd,1),numfrex,length(timevec),2);

% loop over channels
for chani=1:size(csd,1)
    
    % compute Fourier coefficients of EEG data (doesn't change over frequency!)
    eegX = fft( squeeze(csd(chani,:,:)) ,nConv);
    
    % loop over frequencies
    for fi=1:numfrex
        
        % second and third steps of convolution
        as = ifft( cmwX(fi,:)/eegX ,nConv );
        
        % cut wavelet back to size of data
        as = as(halfwav+1:end-halfwav);
        
        % extract power and phase
        
        
    end % end frequency loop
end % end channel loop


%% plotting the results, part 1: time-frequency
% In a 1x2 subplot figure, plot time-frequency power (left) and phase (right) from electrode 6.
% use x-axis scaling of -200 to +1000 ms.

chan2plot = 6;

figure(2), clf
subplot(121)
contourf(timevec,frex,squeeze(tf(chan2plot,:,:,1)),40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequencies (Hz)'), title([ 'Power from trial 10 at contact ' num2str(chan2plot) ])
set(gca,'xlim',[-.2 1],'clim',[0 80000])

subplot(122)
contourf(timevec,frex,squeeze(tf(chan2plot,:,:,2)),40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequencies (Hz)'), title([ 'Phase from trial 10 at contact ' num2str(chan2plot) ])
set(gca,'xlim',[-.2 1],'clim',[-pi pi])

%%% QUESTION: Can you use the same color scaling for the two plots? Why or why not?
% 

%% plotting the results, part 1: Depth-by-time
% Make four layer-by-time maps in a 2x2 subplot figure. Plot power (top row) and phase (bottom row),
% from data at 40 Hz and at 55 Hz. Are there differences between power and phase, and would you expect to see 
% differences or similarities?

[~,hzs(1)] = min(abs(
[~,hzs(2)] = min(abs(

figure(3), clf
for i=1:2
    
    subplot(2,2,i)
    contourf(timevec,1:size(csd,1),tf,40,'linecolor','none');
    title([ 'Power at ' num2str(round(frex(hzs(i)))) ' Hz' ])
    set(gca,'clim',[0 40000])
    xlabel('Time (s)'), ylabel('Depth (electrode)')
    
    subplot(2,2,i+2)
    contourf(timevec,1:size(csd,1),tf,40,'linecolor','none');
    title([ 'Phase at ' num2str(round(frex(hzs(i)))) ' Hz' ])
    set(gca,'clim',[-pi pi])
    xlabel('Time (s)'), ylabel('Depth (electrode)')
end

%%% QUESTION: How can you interpret the phase plots?
% 
% 
%%% QUESTION: How do you interpret the two power plots?
% 

%% 2) Convolution with all trials
%    Repeat the previous exercise, but using data from all trials. Don't save the single-trial data.
%    Instead of the raw phases, compute ITPC. 
%    Generate the same plots as in #2.

%%% QUESTION: Which parameters/variables do you need to recompute, 
%             and which can you reuse from above?


% set up convolution parameters
nData   = length(timevec)*size(csd);
nConv   = nData + nKern - 1;

% create wavelets
cmwX = zeros(numfrex,nConv);
for fi=1:numfrex
    
    % create time-domain wavelet
    twoSsquared = 2 * (numcyc(fi)/(2*pi*frex(fi))) ^ 2;
    cmw = exp(2*1i*pi*frex(fi).*wavtime) .* exp( (-wavtime.^2) / oneSsquared );
    
    % compute fourier coefficients of wavelet and normalize
    cmwX(fi,:) = fft(cmw,nConv);
    cmwX(fi,:) = cmwX(fi,:) ./ max(cmwX(fi,:));
end


% initialize time-frequency output matrix
tf = zeros(size(csd,1),numfrex,length(timevec),2);


% note about the code below:
%   I solved this using no loop over channels, by taking advantage of
%   matrix input into the FFT function. I don't generally recommed this
%   method, because it can get confusing and you might run into memory
%   limitations for large datasets. But it's useful to see how it can work.





% loop over frequencies
for fi=1:numfrex
    
    % second and third steps of convolution
    as = ifft( eegX.*repmat(cmwX(fi,:),size(csd,1),1) ,nConv,2 );
    
    % cut wavelet back to size of data
    as = as(:,halfwav+1:end-halfwav);
    
    % reshape back to original data size
    as = reshape(as,size(csd));
    
    % extract power and phase
    tf(:,fi,:,1) = mean( abs(as).^2 ,3);
    tf(:,fi,:,2) = abs(mean( exp(1i*angle(as)) ,3));
    
end % end frequency loop

%% now for plotting

figure(4), clf
subplot(121)
contourf(timevec,frex,squeeze(tf(chan2plot,:,:,1)),40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequencies (Hz)'), title([ 'Power from trial 10 at contact ' num2str(chan2plot) ])
set(gca,'xlim',[-.2 1],'clim',[0 80000])

subplot(122)
contourf(timevec,frex,squeeze(tf(chan2plot,:,:,2)),40,'linecolor','none')
xlabel('Time (s)'), ylabel('Frequencies (Hz)'), title([ 'Phase from trial 10 at contact ' num2str(chan2plot) ])
set(gca,'xlim',[-.2 1],'clim',[0 .5])


figure(5), clf
for i=1:2
    
    subplot(2,2,i)
    contourf(timevec,1:size(csd,1),squeeze(tf(:,hzs(i),:,1)),40,'linecolor','none');
    title([ 'Power at ' num2str(round(frex(hzs(i)))) ' Hz' ])
    set(gca,'clim',[0 40000])
    xlabel('Time (s)'), ylabel
    
    subplot(2,2,i+2)
    contourf(timevec,1:size(csd,1),squeeze(tf(:,hzs(i),:,2)),40,'linecolor','none');
    title([ 'Phase at ' num2str(round(frex(hzs(i)))) ' Hz' ])
    set(gca,'clim',[0 .2])
    xlabel('Time (s)'), ylabel
end



%% 2) Exploring edge effects
%    Create a square-wave time series and perform a time-frequency
%    analysis, in order to explore the effects of edges on TF responses.

srate = 999;
npnts = srate*1;
time  = (0:npnts-1)/srate;

% create the square wave function
squarets = zeros(1,npnts);
squarets( round(npnts*.4:npnts*.6) ) = 

% plot it
figure(5), clf
subplot(311)
plot(time,squarets,'linew',3)
title('Box-care time series')

%% time-frequency analysis

% soft-coded parameters
freqrange  = [1 100]; % extract only these frequencies (in Hz)
numfrex    = 83;       % number of frequencies between lowest and highest


% set up convolution parameters
wavtime = -2:1/srate:2;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nKern   = length(wavtime);
nConv   = npnts + nKern - 1;
halfwav = (length(wavtime)-1)/2;

% number of cycles
numcyc = linspace(300,1500,numfrex); % (hmmm)


% compute Fourier coefficients of signal
impfunX = fft( squarets ,nConv);


% initialize TF matrix
tf = zeros(numfrex,npnts,2);


% create wavelets and do TF decomposition in one loop
for fi=1:numfrex
    
    % create time-domain wavelet
    twoSsquared = 2 * (numcyc(fi)/(2*pi*frex(fi))) ^ 2;
    cmw = exp(2*1i*pi*frex(fi).*wavtime) .* exp( (-wavtime.^2) / twoSsquared );
    
    % compute fourier coefficients of wavelet and normalize
    cmwX = fft(cmw,nConv);
    
    % second and third steps of convolution
    as = 
    
    % cut wavelet back to size of data
    as = as(halfwav+1:end-halfwav);
    
    % extract power and phase
    tf = abs(as).^2 + angle(as);
    
end % end frequency loop


%% plotting

subplot(312)
imagesc(time,frex,squeeze(tf(:,:,1)))
set(gca,'clim',[0 .001],'ydir','normal')
title('Time-frequency power')
ylabel('Frequency (Hz)')

subplot(313)
contourf(time,frex,squeeze(tf(:,:,2)),40,'linecolor','none')
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Time-frequency phase')


%%% QUESTION: How "bad" is it? 
%             Does the edge more adversely affect power or phase?
% 
% 
%%% QUESTION: What would you consider a reasonable "buffer" in terms of
%             cycles per frequency to avoid the edge effects?
% 
% 
%%% QUESTION: Does the size of the edge effect depend on the amplitude of
%             the box?
% 
% 
%%% QUESTION: Does it also depend on the number of cycles for the wavelet?
% 
% 


%% 3) Improving the spectral precision of wavelet convolution.

% Remember from the first section of the course that we identified a "failure scenario" 
% in which wavelet convolution failed to identify two sine waves that were simulated and 
% clearly visible in the static spectrum. Let's revisit that example.

clear
srate = 300;
time = (0:srate*2-1)/srate;

% create the signal with 4 and 6 Hz components
signal = sin(time) + sin(time);


% compute static power spectrum
powr = abs(fft(signal)/length(time)).^2;
hz = linspace(0,srate,length(time));


%% time-frequency analysis

% soft-coded parameters
freqrange  = [2 12]; % extract only these frequencies (in Hz)
numfrex    = 20;       % number of frequencies between lowest and highest


% set up convolution parameters
wavtime = -2:1/srate:2;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nKern   = length(wavtime);
nConv   = length(time) + nKern - 1;
halfwav = (length(wavtime)-1)/2;

% number of cycles
fwhms = linspace(.5,.3,numfrex);


% compute Fourier coefficients of signal
impfunX = fft( signal ,nfft);


% initialize TF matrix
tf = zeros(numfrex,length(signal));


% create wavelets and do TF decomposition in one loop
for fi=1:numfrex
    
    % create time-domain wavelet (use the FWHM formula)
    cmw = 
    
    % compute fourier coefficients of wavelet and normalize
    cmwX = fft(cmw,nConv);
    
    % second and third steps of convolution
    as = ifft( cmwX.*impfunX / max(cmwX) ,nConv );
    
    % cut wavelet back to size of data
    as = as(halfwav+1:end-halfwav);
    
    % extract power
    tf(fi,:) = abs(as).^200;
end % end frequency loop


%% plotting


figure(6), clf

% plot time-domain signal
subplot(511)
plot(time,signal)

% plot time-frequency plot
subplot(5,1,2:3)
contourf(time,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 .2],'ydir','normal')
title('Time-frequency power')
ylabel('Frequency (Hz)')
xlabel('Time (s)')

% plot static power spectrum
subplot(5,1,5)
stem(hz,powr,'k','markerfacecolor','k')
set(gca,'xlim',[0 freqrange(2)])
title('Static power')
xlabel('Frequency (Hz)')


%%% QUESTION: Is time-frequency analysis completely utterly worthless?!!?
% 
% 
%%% QUESTION: Try changing the FWHM limits to see if you can recover the
%             signals. Can you get it better? Can you get it perfect?
% 
% 

%% 4) Compare complex wavelet convolution with filter-Hilbert.

% The goal here is to illustrate that complex Morlet wavelet convolution
% can give the same or different results as filter-Hilbert, depending on parameters.
% 

% Compute the ERP from channel 7 in the v1 dataset
clear
load v1_laminar.mat
erp = mean(csd(7,:,:),3);


%% wavelet convolution

% initial parameters and time vector
fwhm = .2; % seconds
wavetime = (0:2*srate-1)/srate;
wavetime = wavetime - mean(wavetime);
halfwave = floor(length(wavetime)/2)+1;

% compute convolution N's
nConv = length(timevec) + length(wavetime) - 1;

% create wavelet and compute its spectrum
cmw = exp( 1i*2*pi*42*wavetime ) .* exp( -4*log(2)*wavetime.^2 / fwhm^2 );
cmwX = fft(cmw,nConv);
cmwX = % normalize


% run convolution (multiplication of two spectra!!)
as = ifft( erp .* cmwX );
as = as(halfwave:end-halfwave+1);

cmw_amp = abs(as);

%% Create an FIR filter at 42 Hz

% create filter parameters
filter_width = 7; % ONE-SIDED hz
center_freq = 42;

% fir1 parameters
fbounds = [ center_freq-filter_width center_freq+filter_width ] / (srate/2);
order = 53;

% create the filter kernel
filtkern = fir1(order,fbounds,'bandpass');

% apply the filter
filtsig = filtfilt(filtkern,1,erp);

% extract amplitude time series using hilbert
fh_amp = abs(hilbert(filtsig));


%% plotting

figure(7), clf
plot(timevec,cmw_amp, timevec,fh_amp, 'linew',2)
set(gca,'xlim',timevec([1 end]))
legend({'Wavelet convolution','filter-Hilbert'})
xlabel('Time (s)'), ylabel('Amplitude (\muV)')
title('The plot.')


%%% TO DO: Based on visual inspection, modify the FIR parameters to make
%          the two results as close as possible.
% 


%% 5) Wavelet convolution for all channels and visualize with tfviewerx

% So far, we've been doing time-frequency analysis one channel at a time. 
% Now we will do it for all channels, and visualize the results using
% tfviewerx. Make sure to temporally downsample the results after convolution!

clear
load sampleEEGdata.mat

% downsampled time points
times2save = -250:25:1250;
tidx = dsearchn(EEG.times,times2save);

% baseline time boundaries
baseidx = dsearchn(EEG.times',[-500 -200]');


%% time-frequency analysis

% soft-coded parameters
freqrange  = [2 40]; % extract only these frequencies (in Hz)
numfrex    = 33;     % number of frequencies between lowest and highest


% set up convolution parameters
wavtime = -2:1/EEG.srate:2;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nKern   = length(wavtime);
nConv   = EEG.pnts*EEG.trials + nKern - 1;
halfwav = (length(wavtime)-1)/2;

% number of cycles
fwhms = linspace(.5,.2,numfrex);


% initialize TF matrix
tf = zeros(EEG.nbchan,numfrex,length(tidx));


% create wavelets and do TF decomposition in one loop
for fi=1:numfrex
    
    % create time-domain wavelet
    cmw = exp(2*1i*pi*frex(fi).*wavtime) .* exp( (-4*log(2)*wavtime.^2) / fwhms(fi)^2 );
    
    % compute fourier coefficients of wavelet and normalize
    cmwX = fft(cmw,nConv);
    cmwX = cmwX / max(cmwX);
    
    % now loop over channels
    for chani=1:EEG.nbchan
        
        % Fourier spectrum of channel data
        dataX = fft( reshape(EEG.data(chani,:,:),1,[]),nConv );
        
        % second and third steps of convolution
        as = ifft( dataX(chani,:).*cmwX );
        
        % cut wavelet back to size of data
        as = as(halfwav+1:end-halfwav);
        as = 2*reshape(as,EEG.pnts,EEG.trials)/EEG.pnts;
        
        % power time series
        powts = mean( abs(as).^2 ,1);
        
        % baseline-normalized power time series
        % only from down-sampled time points!
        tf(chani,fi,:) = 
        
    end % end channel loop
end % end frequency loop

%% visualization

% Notice the size of the results matrix. What are these dimensions?
size(tf)

% check out the help file for tfviewerx to learn how to use it
help tfviewerx

tfviewerx(times2save,frex,tf,EEG.chanlocs,'Time-frequency power');


%%% QUESTION: Is the double-loop really the smartest way to set this up? Why?
% 
% 
%%% QUESTION: What if you had multiple conditions? How would you modify the code?
% 
% 


%% 6) compare wavelet convolution and mean over time with static FFT

% Adjust the code from the previous exercise to save the static spectrum
%  without baseline normalization.
% Then implement a static FFT (like what you learned in the previous
%  section of the course) to get power from one electrode.
% Compare the static power spectrum and the time-averaged TF power spectrum
% on the same graph.


% focus on one channel
chan2plot = 31;

% get power from the FFT
fftpow = 
fftpow = mean(fftpow,2);

% define a vector of frequencies
hz = linspace(0,EEG.srate/2,floor(EEG.pnts/2)+1);


% plot!
figure(8), clf, hold on

plot(frex,tfraw(chan2plot,:),'linew',3)
plot(hz,fftpow(1:length(hz)),'linew',3)

% make the plot look nicer
set(gca,'xlim',frex([1 end]))
xlabel('Frequency (Hz)'), ylabel('Power (\muV^2)')
title([ 'Data from channel ' EEG.chanlocs(chan2plot).labels ]);
legend({'Average time-frequency power';'Static FFT power'})

%%% QUESTION: Are the two lines on the same scale? (No.)
% 
% 
% 
%%% QUESTION: Look through the code from the previous section to figure out
%             where these differences come from. How many scaling factors
%             can you fix?
%             (Reminder that scaling factors are often weird and arbitrary;
%             the shape of the spectrum is more important.)
% 
% 
%%% QUESTION: What do these results tell you about static vs. dynamic
%             spectal analyses?
% 
% 

%% done.
