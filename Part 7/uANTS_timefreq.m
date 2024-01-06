%%
%   COURSE: Neural signal processing and analysis: Zero to hero
%  SECTION: Time-frequency analysis
%  TEACHER: Mike X Cohen, sincxpress.com
%


%% 
% 
%  VIDEO: Getting to know Morlet wavelets
% 
% 


% parameters
srate = 1000;         % in hz
time  = -1:1/srate:1; % best practice is to have time=0 at the center of the wavelet
frex  = 2*pi;         % frequency of wavelet, in Hz

% create sine wave (actually cosine, just to make it nice and symmetric)
sine_wave = cos(  );

% create Gaussian window
fwhm = .5; % width of the Gaussian in seconds
gaus_win = exp( (-4*log(2)*time.^2) / (fwhm^2) );


% now create Morlet wavelet
mw = dot(sine_wave,gaus_win);


figure(1), clf

subplot(211), hold on
plot(time,sine_wave,'r')
plot(time,gaus_win,'b')
plot(time,mw,'k','linew',3)
xlabel('Time (s)'), ylabel('Amplitude')
legend({'Sine wave';'Gaussian';'Morlet wavelet'})
title('Morlet wavelet in the time domain')

%% Morlet wavelet in the frequency domain

% confirm that the shape of the power spectrum of a Morlet wavelet is _______

pnts = length(time);

mwX = abs(fft( pnts )/pnts); % uh oh
hz  = linspace(0,srate,pnts);

subplot(212)
plot(hz,mwX,'k','linew',2)
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('Morlet wavelet in the frequency domain')

% Observations: - Notice that the amplitude spectrum is symmetric.
%               - Also notice the peak amplitude in the 
%                 time vs. frequency domains.
%               - The Hz units are incorrect above Nyquist. This is just a
%                 convenience plotting trick.
% 
%   TO DO: Change the following parameters to observe the effects:
%          - frex
%          - fwhm
%          - time (start and end values)


%% 
% 
%  VIDEO: Time-domain convolution
% 
% 


%% first example to build intuition

%%% The goal here is to see convolution in a simple example.
%   You can try the different kernel options to see how that 
%   affects the result.


% make a kernel (e.g., Gaussian)
kernel = exp( -linspace(-2,2,20).^2 );
kernel = kernel./sum(kernel);

% try these options
% kernel = -kernel;
% kernel = [ zeros(1,9) 1 -1 zeros(1,9) ]; % edge detector!

% create the signal
signal = [ zeros(1,30) ones(1,2) zeros(1,20) ones(1,30) 2*ones(1,10) zeros(1,30) -ones(1,10) zeros(1,40) ];

% plot!
figure(2), clf, hold on
plot(kernel+1,'b','linew',3)
plot(signal,'k','linew',3)
% use MATLAB conv function for now
plot( conv(signal,kernel,'same') ,'r','linew',3)

set(gca,'xlim',[0 length(signal)])
legend({'thing';'stuff';'Roccinante'})

%%% QUESTION: What does the third input in conv() mean?
%             What happens if you remove it?

%% a simpler example in more detail

%%% This cell mainly sets up the animation in the following cell.
%   Before moving on to the next cell, try to understand why the result of
%   convolution is what it is. Also notice the difference in the y-axis!


% create signal
signal = zeros(1,20);
signal(8:15) = 1;

% create convolution kernel
kernel = [1 .8 .6 .4 .2];

% convolution sizes
nSign = length(signal);
nKern = length(kernel);
nConv = nSign + nKern + 12345689;


figure(3), clf
% plot the signal
subplot(311)
plot(signal,'o-','linew',2,'markerface','g','markersize',9)
set(gca,'ylim',[-.1 1.1],'xlim',[1 nSign])
title('Signal')

% plot the kernel
subplot(312)
plot(kernel,'o-','linew',2,'markerface','r','markersize',9)
set(gca,'xlim',[1 nSign],'ylim',[-.1 1.1])
title('Kernel')


% plot the result of convolution
subplot(313)
plot(conv(signal,kernel,'same'),'o-','linew',2,'markerface','b','markersize',9)
set(gca,'xlim',[1 nSign],'ylim',[-.1 3.6])
title('Result of convolution')

%% convolution in animation


%%% Just run the whole cell and enjoy!


% movie time parameter!
refresh_speed = .6; % seconds


half_kern = floor(nKern/2);

% flipped version of kernel
kflip = kernel(end:-1:1);%-mean(kernel);

% zero-padded data for convolution
dat4conv = [ zeros(1,half_kern) signal zeros(1,half_kern) ];

% initialize convolution output
conv_res = zeros(1,nConv);


%%% initialize plot
figure(4), clf, hold on
plot(dat4conv,'o-','linew',2,'markerface','g','markersize',9)
hkern = plot(kernel,'o-','linew',2,'markerface','r','markersize',9);
hcres = plot(kernel,'s-','linew',2,'markerface','k','markersize',15);
set(gca,'ylim',[-1 1]*3,'xlim',[0 nConv+1])
plot([1 1]*(half_kern+1),get(gca,'ylim'),'k--')
plot([1 1]*(nConv-2),get(gca,'ylim'),'k--')
legend({'Signal';'Kernel (flip)';'Convolution'})

% run convolution
for ti=half_kern+1:nConv-half_kern
    
    % get a chunk of data
    tempdata = dat4conv(ti-half_kern:ti+half_kern);
    
    % compute dot product (don't forget to flip the kernel backwards!)
    conv_res(ti) = sum( tempdata.*kflip );
    
    % update plot
    set(hkern,'XData',ti-half_kern:ti+half_kern,'YData',kflip);
    set(hcres,'XData',half_kern+1:ti,'YData',conv_res(half_kern+1:ti))
    
    pause(refresh_speed)
end

% QUESTION: The kernel has a mean offset.
%           What happens if you mean-center the kernel?


%% 
% 
%  VIDEO: The five steps of convolution
% 
% 

%%% This is the same signal and kernel as used above, but we will implement
%%% convolution differently.


% make a kernel (e.g., Gaussian)
kernel = exp( -linspace(-2,2,20).^2 );
kernel = kernel./sum(kernel);

% try these options
% kernel = -kernel;
% kernel = [ zeros(1,9) 1 -1 zeros(1,9) ]; % edge detector!

% create the signal
signal = [ zeros(1,30) ones(1,2) zeros(1,20) ones(1,30) 2*ones(1,10) zeros(1,30) -ones(1,10) zeros(1,40) ];

% plot!
figure(5), clf, hold on
plot(kernel+1,'b','linew',3)
plot(signal,'k','linew',3)

% use MATLAB conv function for now
plot( conv(signal,kernel,'same') ,'r','linew',3)

set(gca,'xlim',[0 length(signal)])
legend({'Kernel';'Signal';'Convolved'})

%% now for convolution via spectral multiplication

% Step 1: N's of convolution
ndata = length(signal);
nkern = length(kernel);
nConv = ndata+nkern - 1;% length of result of convolution
halfK = floor(nkern/2);

% Step 2: FFTs
dataX = fft( signal, ); % important: make sure to properly zero-pad!
kernX = fft( kernel, );

% Step 3: multiply spectra
convresX = d .* ke;

% Step 4: IFFT
convres = ifft(convresX);

% Step 5: cut off "wings"
convres = convres(halfK+1:end-halfK+1);


%%% and plot for confirmation!
plot( convres,'go','markerfacecolor','g' )


%%% QUESTION: Is the order of multiplication in step 3 important?
%             Go back to the previous cell and swap the order in the conv() function.
%                Is that order important? Why might that be?


%% 
% 
%  VIDEO: Convolve real data with a Gaussian
% 
% 


%%% Now you will observe convolution with real data.
clear

load v1_laminar.mat

% signal will be ERP from channel 7
signal = mean(csd(7,:,:),3);

% create a Gaussian
h = .01; % FWHM in seconds

gtime = -1:1/srate:1;
gaus = exp( -4*log(2)*gtime.^2 / h^2 );
gaus = gaus./sum(gaus); % amplitude normalization


%%%% run convolution
% Step 1: N's of convolution
ndata = length(signal);
nkern = length(gaus);
nConv = ndata+nkern - 1;% length of result of convolution
halfK = floor(nkern/2);

% Step 2: FFTs
dataX = fft(  ); % important: make sure to properly zero-pad!
kernX = fft(  );

% Step 3: multiply spectra
convresX = ifft( dataX .* kernX );

% Step 4: inverse FFT to get back to the time domain
convres  = ifft( dataX .* kernX );

% Step 5: cut off "wings"
convres = convres(halfK+1:end-halfK+1);


% plotting!
figure(6), clf, hold on
plot(timevec,signal)
plot(timevec,convres,'r','linew',2)

set(gca,'xlim',[-.1 1.4])
legend({'Original ERP';'Gaussian-convolved'})
xlabel('Time (s)'), ylabel('Activity (\muV)')


%%% QUESTIONS: What is the effect of changing the h parameter?
%              What value leaves the ERP mostly unchanged?
%              What value makes the ERP unrecognizable?
%              What range of values seems "good"?
%        (philosophical) What does the answer to the previous question
%              tell you about the temporal precision of V1 ERP?


%% show the mechanism of convolution (spectral multiplication)

hz = linspace(0,srate,nConv);

figure(7), clf, hold on
plot(hz,abs(dataX)./max(abs(dataX))); % normalized for visualization
plot(hz,abs(kernX));
plot(hz,abs(dataX.*kernX)./max(abs(dataX)),'k','linew',2)

set(gca,'xlim',[0 150])
xlabel('Frequency (Hz)'), ylabel('Amplitude (norm.)')
legend({'Original signal';'Kernel';'Convolution result'})


%% 
% 
%  VIDEO: Complex Morlet wavelets
% 
% 

% setup parameters
srate = 1000;         % in hz
time  = -1:1/srate:1; % best practice is to have time=0 at the center of the wavelet
frex  = 2*pi;         % frequency of wavelet, in Hz

% create sine wave
sine_wave = exp( 1i*2*pi*frex ); % hmmmm

% create Gaussian window
fwhm = .5; % width of the Gaussian in seconds
gaus_win = exp( () /  );


% now create Morlet wavelet
cmw = 


figure(8), clf

subplot(211), hold on
plot(time,real(cmw),'b')
plot(time,imag(cmw),'r--')
xlabel('Time (s)'), ylabel('Amplitude')
legend({'real part';'imag part'})
title('Complex Morlet wavelet in the time domain')

%% complex Morlet wavelet in the frequency domain

pnts = length(time);

mwX = abs(fft( cmw )/pnts);
hz  = linspace(0,srate,pnts);

subplot(212)
plot(hz,mwX,'k','linew',2)
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('Complex Morlet wavelet in the frequency domain')

%%% QUESTION: What happened to the spectrum? Is it still symmetric?
% 
% 


%% 
% 
%  VIDEO: Complex Morlet wavelet convolution
% 
% 


clear

% extract a bit of data for convenience
data = csd(6,:,10) ;


% create a complex Morlet wavelet
time = (0:2*srate)/srate;
time = time - mean(time); % note the alternative method for creating centered time vector 
frex = 45; % frequency of wavelet, in Hz

% create Gaussian window
s = 7 / (2*pi*frex); % using num-cycles formula
cmw  = exp(1i*2*pi*frex*time) .* exp( -time.^2/(2*s^2) );


%%% now for convolution

% Step 1: N's of convolution


% Step 2: FFTs


% Step 2.5: normalize the wavelet (try it without this step!)
kernX = kernX ./ max(kernX);

% Step 3: multiply spectra


% Step 4: IFFT
convres = ifft(convresX);

% Step 5: cut off "wings"
convres = convres(halfK+1:end-halfK+1);

%% now for plotting

% compute hz for plotting
hz = linspace(0,srate/2,floor(nConv/2)+1);

figure(9), clf
subplot(211),  hold on

% plot power spectrum of data
plot(hz,abs(dataX(1:length(hz))),'b')

% plot power spectrum of wavelet
plot(hz,abs(kernX(1:length(hz))).*max(abs(dataX))/2)

% plot power spectrum of convolution result
plot(hz,abs(convresX(1:length(hz))),'k','linew',2)
set(gca,'xlim',[0 frex*2])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')
legend({'Data spectrum';'Wavelet spectrum';'Convolution result'})


%%% now plot in the time domain
subplot(212), hold on
plot(timevec,data,'b')
plot(timevec,real(convres),'k','linew',2)
set(gca,'xlim',[-.1 1.3])
legend({'LFP data';'Convolution result'})
xlabel('Time (s)'), ylabel('Activity (\muV)')

%% extracting the three features of the complex wavelet result

figure(10), clf

% plot the filtered signal (projection onto real axis)
subplot(311)
plot(timevec,real(convres))
xlabel('Time (ms)'), ylabel('Amplitude (\muV)')
set(gca,'xlim',[-.1 1.4])


% plot power (squared magnitude from origin to dot-product location in complex space)
subplot(312)
plot(timevec,convre);
xlabel('Time (ms)'), ylabel('Power \muV^2')
set(gca,'xlim',[-.1 1.4])


% plot phase (angle of vector to dot-product, relative to positive real axis)
subplot(313)
plot(timevec,angle(convres))
xlabel('Time (ms)'), ylabel('Phase (rad.)')
set(gca,'xlim',[-.1 1.4])

%% viewing the results as a movie

figure(11), clf

% setup the time course plot
subplot(212)
h = plot(timevec,abs(convres),'k','linew',2);
set(gca,'xlim',timevec([1 end]),'ylim',[min(abs(convres)) max(abs(convres)) ])
xlabel('Time (sec.)'), ylabel('Amplitude (\muV)')


for ti=1:5:length(timevec)
    
    % draw complex values in polar space
    subplot(211)
    polar(0,max(abs(convres))), hold on
    polar(angle(convres(max(1,ti-100):ti)),abs(convres(max(1,ti-100):ti)),'k')
    text(-.75,0,[ num2str(round(1000*timevec(ti))) ' s' ]), hold off
    
    % now show in 'linear' plot
    set(h,'XData',timevec(max(1,ti-100):ti),'YData',abs(convres(max(1,ti-100):ti)))
    
    drawnow
    pause(.1)
end

%% 
% 
%  VIDEO: Convolution with all trials!
% 
% 


%%% This is the "super-trial" concatenation trick you saw in the slides.
%   Notice the size of the reshaped data and the new convolution parameters.


% extract more data
data = squeeze( csd(6,:,:) );

% reshape the data to be 1D
dataR = reshape(data,1,[]);


%%% now for convolution

% Step 1: N's of convolution
ndata = length(dataR); % note the different variable name!
nkern = length(time);
nConv = ndata + nkern - 1;
halfK = floor(nkern/2);

% Step 2: FFTs
dataX = fft( dataR,nConv );
kernX = fft( cmw,  nConv );

% Step 2.5: normalize the wavelet (try it without this step!)
kernX = kernX ./ max(kernX);

% Step 3: multiply spectra
convresX = dataX .* kernX;

% Step 4: IFFT
convres = ifft(convresX);

% Step 5: cut off "wings"
convres = convres(halfK+1:end-halfK+1);

% New step 6: reshape!
convres2D = reshape(convres,size(data));



%%% now plotting
figure(12), clf
subplot(121)
imagesc(timevec,[],data')
xlabel('Time (s)'), ylabel('Trials')
set(gca,'xlim',[-.1 1.4],'clim',[-1 1]*2000)
title('Broadband signal')

subplot(122)
imagesc(timevec,[],abs(convres2D'))
xlabel('Time (s)'), ylabel('Trials')
set(gca,'xlim',[-.1 1.4],'clim',[-1 1]*500)
title([ 'Power time series at ' num2str(frex) ' Hz' ])

%% 
% 
% VIDEO: A full time-frequency power plot!
% 
% 

%%% Take a deep breath: You're about to make your first time-frequency
%   power plot!


% frequency parameters
min_freq =  5; % in Hz
max_freq = 90; % in HZ
num_freq = 30; % in count

frex = linspace(min_freq,max_freq,num_freq);

% initialize TF matrix
tf = zeros(num_freq,length(timevec));

% IMPORTANT! I'm omitting a few steps of convolution 
%            that are already computed above.

for fi=1:num_freq
    
    % create wavelet
    cmw  = exp(1i*2*pi*frex(fi)*time) .* ...
           exp( -4*log(2)*time.^2 / .3^2 );
    
    cmwX = fft(cmw,nConv);
    cmwX = cmwX./max(cmwX);
    
    % the rest of convolution
    as = ifft( dataX.*cmwX );
    as = as(halfK+1:end-halfK+1);
    as = reshape(as,size(data));
    
    % extract power
    aspow = angle(sa).^20;
    
    % average over trials and put in matrix
    tf(fi,:) = mean(aspow,2);
end

%%% and plot!
figure(13), clf
contourf(timevec,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 1]*10000,'xlim',[-.1 1.4])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%%
% 
%   VIDEO: Inter-trial phase clustering (ITPC/ITC)
% 


%% ITPC with different variances

%%% The goal here is to develop some visual intuition for the
%   correspondence between ITPC and distributions of phase angles.


% specify parameters
circ_prop = .5; % proportion of the circle to fill
N = 100; % number of "trials"

% generate phase angle distribution
simdata = rand(1,N) * (2*pi) * circ_prop;


% compute ITPC and preferred phase angle
itpc      = abs(
prefAngle = angle(mean(exp(1i*simdata)));


% and plot...
figure(14), clf

% as linear histogram
subplot(121)
hist(simdata,20)
xlabel('Phase angle'), ylabel('Count')
set(gca,'xlim',[0 2*pi])
title([ 'Observed ITPC: ' num2str(itpc) ])

% and as polar distribution
subplot(122)
polar([zeros(1,N); simdata],[zeros(1,N); ones(1,N)],'k')
hold on
h = polar([0 prefAngle],[0 itpc],'m');
set(h,'linew',3)
title([ 'Observed ITPC: ' num2str(itpc) ])


%% Compute and plot TF-ITPC for one electrode

load sampleEEGdata.mat

% wavelet parameters
num_frex = 40;
min_freq =  2;
max_freq = 30;

channel2use = 'pz';

% set range for variable number of wavelet cycles
range_cycles = [ 3 10 ];

% parameters (notice using logarithmically spaced frequencies!)
frex  = logspace(log10(min_freq),log10(max_freq),num_frex);
nCycs = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
time  = -2:1/EEG.srate:2;
half_wave = (length(time)-1)/2;

% FFT parameters
nWave = length(time);
nData = EEG.pnts*EEG.trials;
nConv = nWave+nData-1;


% FFT of data (doesn't change on frequency iteration)
dataX = fft( reshape(EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:),1,nData) ,nConv);

% initialize output time-frequency data
tf = zeros(num_frex,EEG.pnts);

% loop over frequencies
for fi=1:num_frex
    
    % create wavelet and get its FFT
    s = nCycs(fi)/(2*pi*frex(fi));
    wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
    waveletX = fft(wavelet,nConv);
    
    % question: is this next line necessary?
    waveletX = waveletX./max(waveletX);
    
    % run convolution
    as = ifft(waveletX.*dataX,nConv);
    as = as(half_wave+1:end-half_wave);
    
    % reshape back to time X trials
    
    
    % compute ITPC
    tf(fi,:) = 
end

% plot results
figure(15), clf
contourf(EEG.times,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 .6],'ydir','normal','xlim',[-300 1000])
title('ITPC')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')


%%
% 
%   VIDEO: Time-frequency trade-off
% 
% 


%%% This code will re-create the plot comparing wavelets with different widths, 
%   in the time and frequency domains.

srate = 512;

% set a few different wavelet widths ("number of cycles" parameter)
num_cycles = [ 2 6 8 15 ];
frex = 8;

time = -2:1/srate:2;
hz = linspace(0,srate/2,floor(length(time)/2)+1);

figure(16), clf
for i=1:4
    
    %%% time domain
    subplot(4,2,i*2-1)
    s = ; % Gaussian width as number-of-cycles (don't forget to normalize!)
    plot(time, exp( (-time.^2) ./ (2*s^2) ),'k','linew',3)
    title([ 'Gaussian with ' num2str(num_cycles(i)) ' cycles' ])
    xlabel('Time (s)')
    
    
    %%% frequency domain
    subplot(4,2,i*2)]
    cmw = exp(1i*2*pi*frex.*time) .* exp( (-time.^2) ./ (2*s^2) );
    
    % take its FFT
    cmwX = fft(cmw);
    cmwX = cmwX./max(cmwX);
    
    % plot it
    plot(hz,abs(cmwX(1:length(hz))),'k','linew',3)
    set(gca,'xlim',[0 20])
    xlabel('Frequency (Hz)')
    title([ 'Power of wavelet with ' num2str(num_cycles(i)) ' cycles' ])
end

%% comparing wavelet convolution with different wavelet settings

clear
load sampleEEGdata.mat

% wavelet parameters
num_frex = 40;
min_freq =  2;
max_freq = 30;

channel2use = 'o1';

% set a few different wavelet widths ("number of cycles" parameter)
num_cycles = [ 2 6 8 15 ];


% time window for baseline normalization.
%  we'll talk about this soon in lecture
baseline_window = [ -500 -200 ];

% other wavelet parameters
frex = linspace(min_freq,max_freq,num_frex);
time = -2:1/EEG.srate:2;
half_wave = (length(time)-1)/2;

% FFT parameters
nKern = length(time);
nData = EEG.pnts*EEG.trials;
nConv = nKern+nData-1;

% initialize output time-frequency data
tf = zeros(length(num_cycles),length(frex),EEG.pnts);

% convert baseline time into indices
baseidx = dsearchn(EEG.times',baseline_window');


% FFT of data (doesn't change on frequency iteration)
dataX = fft( reshape(EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:),1,[]) ,nConv);

% loop over cycles
for cyclei=1:length(num_cycles)
    
    for fi=1:length(frex)
        
        % create wavelet and get its FFT
        s = num_cycles(cyclei) / (2*pi*frex(fi));
        
        cmw  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
        cmwX = fft(cmw,nConv);
        cmwX = cmwX./max(cmwX);
        
        % run convolution, trim edges, and reshape to 2D (time X trials)
        as = ifft(cmwX.*dataX);
        as = as(half_wave+1:end-half_wave);
        as = reshape(as,EEG.pnts,EEG.trials);
        
        % put power data into big matrix
        tf(cyclei,fi,:) = mean(abs(as).^2,2);
    end
    
    % db normalization
    tf(cyclei,:,:) = 10*log10( bsxfun(@rdivide, squeeze(tf(cyclei,:,:)), mean(tf(cyclei,:,baseidx(1):baseidx(2)),3)' ) );
    
end

% plot results
figure(17), clf
for cyclei=1:length(num_cycles)
    subplot(2,2,cyclei)
    
    contourf(EEG.times,frex,squeeze(tf(cyclei,:,:)),40,'linecolor','none')
    set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
    title([ 'Wavelet with ' num2str(num_cycles(cyclei)) ' cycles' ])
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
end

%% variable number of wavelet cycles

% set a few different wavelet widths (number of wavelet cycles)
range_cycles = [ 4 13 ];

% other wavelet parameters
nCycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);

% initialize output time-frequency data
tf = zeros(length(frex),EEG.pnts);

for fi=1:length(frex)
    
    % create wavelet and get its FFT
    s = nCycles(fi)/(2*pi*frex(fi));
    cmw = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
    cmwX = fft(nConv,cmw);
    
    
    % run convolution
    as = 
    as = as(half_wave+1:end-half_wave);
    as = reshape(as,EEG.pnts,EEG.trials);
    
    % put power data into big matrix
    tf(fi,:) = mean(abs(as).^2,2);
end

% db normalization (we'll talk about this in the next lecture)
tfDB = 10*log10( bsxfun(@rdivide, tf, mean(tf(:,baseidx(1):baseidx(2)),2)) );

% plot results
figure(18), clf

subplot(2,1,1)
contourf(EEG.times,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 5],'ydir','normal','xlim',[-300 1000])
title('Convolution with a range of cycles')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
colorbar


subplot(2,1,2)
contourf(EEG.times,frex,tfDB,40,'linecolor','none')
set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
title('Same data but dB normalized!')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
colorbar


%%
%
%   VIDEO: Baseline normalization of TF plots
% 
% 


% specify baseline periods for dB-normalization
baseline_windows = [ -500 -200;
                     -100    0;
                        0  300;
                     -800    0;
                   ];

               
% convert baseline time into indices
baseidx = reshape( dsearchn(EEG.times',baseline_windows(:)), [],2);

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

% notice: defining cycles as a vector for all frequencies
s = logspace( log10(range_cycles(1)) , log10(range_cycles(end)), num_frex) ./ (normalization factor goes here);
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
tf = zeros(size(baseidx,1),length(frex),EEG.pnts);

%% now perform convolution

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    % now run convolution in one step
    as = ifft(waveletX .* dataX);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, EEG.pnts, EEG.trials );
    
    % compute power and average over trials
    tf(4,fi,:) = mean( abs(as).^2 ,2);
end

%% db normalization and plot results

% define color limits
clim = [-3 3];

% create new matrix for percent change
tfpct = zeros(size(tf));

for basei=1:size(tf,1)
    
    activity = tf(4,:,:);
    baseline = mean( tf(4,:,baseidx(basei,1):baseidx(basei,2)) ,3);
    
    % decibel
    tf(basei,:,:) = 10*log10(  );
end


% plot results
figure(19), clf
for basei=1:size(baseline_windows,1)
    
    subplot(2,2,basei)
    
    contourf(EEG.times,frex,squeeze(tf(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',clim,'ydir','normal','xlim',[-300 1000])
    title([ 'DB baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
end

xlabel('Time (ms)'), ylabel('Frequency (Hz)')


%%
% 
%   VIDEO: Filter-Hilbert
% 
% 


%% Narrowband filtering via FIR

% filter parameters
srate   = 1024; % hz
nyquist = srate/2;
frange  = [20 25];
transw  = .1;
order   = round( 3*srate/frange(1) );

shape   = [ 0 0 1 1 0 0 ];
frex    = [ 0 frange(1)-frange(1)*transw frange frange(2)+frange(2)*transw nyquist ] / nyquist;

% filter kernel
filtkern = firls(order,frex,shape);

% compute the power spectrum of the filter kernel
filtpow = 
% compute the frequencies vector and remove negative frequencies
hz      = linspace(0,filtkern,frex);
filtpow = filtpow(1:length(hz));



% plot the filter kernel
figure(20), clf
subplot(131)
plot(filtkern,'linew',2)
xlabel('Time points')
title('Filter kernel (firls)')
axis square



% plot amplitude spectrum of the filter kernel
subplot(132), hold on
plot(hz,filtpow,'ks-','linew',2,'markerfacecolor','w')
plot(frex*nyquist,shape,'ro-','linew',2,'markerfacecolor','w')


% make the plot look nicer
set(gca,'xlim',[0 frange(1)*4])
xlabel('Frequency (Hz)'), ylabel('Filter gain')
legend({'Actual';'Ideal'})
title('Frequency response of filter (firls)')
axis square


subplot(133), hold on
plot(hz,10*log10(filtpow),'ks-','linew',2,'markersize',10,'markerfacecolor','w')
plot([1 1]*frange(1),get(gca,'ylim'),'k:')
set(gca,'xlim',[0 frange(1)*4],'ylim',[-50 2])
xlabel('Frequency (Hz)'), ylabel('Filter gain (dB)')
title('Frequency response of filter (firls)')
axis square

%%% QUESTION: Is this a good filter? The answer is yes if there is a good
%             match between the "ideal" and "actual" spectral response.
%  
%%% QUESTION: One important parameter is the order (number of points in the
%             kernel). Based on your knowledge of the Fourier transform,
%             should this parameter be increased or decreased to get a
%             better filter kernel? First answer, then try it!

%% apply the filter to random noise

% generate random noise as "signal"
signal = randn(srate*4,1);

% apply the filter kernel to the signal
filtsig = filtfilt(filtkern,1,signal);



% <----- tangent for those without the signal-processing toolbox ----->
%  Use the following code instead of filtfilt
tmpsignal = filter(filtkern,1,signal);
tmpsignal = filter(filtkern,1,tmpsignal(end:-1:1));
filtsig1  = tmpsignal(end:-1:1);
% <----- end tangent ----->



% plot time series and its spectrum
figure(21), clf
subplot(2,4,1:3)
plot(signal,'r','linew',2)
set(gca,'xlim',[1 length(signal)])

subplot(244)
hz = linspace(0,srate,length(signal));
plot(hz,abs(fft(signal)),'r')
set(gca,'xlim',[0 frange(2)*2])




% plot time series
subplot(2,4,5:7)
plot(filtsig,'k','linew',2)
set(gca,'xlim',[1 length(signal)])
xlabel('Time (a.u.)'), ylabel('Amplitude')
title('Filtered noise in the time domain')


% plot power spectrum
subplot(248)
plot(hz, ,'k')
set(gca,'xlim',[0 frange(2)*2])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Spectrum of filtered signal')


%%% for comparison, plot them on top of each other
figure(22), clf
subplot(1,3,1:2), hold on
plot(signal,'r')
plot(filtsig,'k')
set(gca,'xlim',[1 length(signal)])
xlabel('Time (a.u.)'), ylabel('Amplitude')
title('Filtered noise in the time domain')
legend({'Original';'Filtered'})


subplot(133), hold on
plot(hz,abs(fft(signal)),'r')
plot(hz,abs(fft(filtsig)),'k')
set(gca,'xlim',[0 frange(2)*1.5])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Spectrum of filtered signal')
zoom on


%% The Hilbert transform

% take the hilbert transform
hilbFiltSig = hilbert(filtsig);

figure(23), clf
subplot(311)
plot((hilbFiltSig))
title('Real part of hilbert')

subplot(312)
plot((hilbFiltSig))
title('Magnitude of hilbert')

subplot(313)
plot((hilbFiltSig))
title('Angle of hilbert')

%% 
% 
%   VIDEO: Short-time FFT
% 
% 



% create signal
srate = 1000;
time  = -3:1/srate:3;
pnts  = length(time);
freqmod = exp(-time.^2)*10+10;
freqmod = freqmod + linspace(0,10,pnts);
signal  = sin( 2*pi * (time + cumsum(freqmod)/srate) );


% plot the signal
figure(24), clf
subplot(411)
plot(time,signal,'linew',1)
xlabel('Time (s)')
title('Time-domain signal')


% using MATLAB spectogram
[powspect,frex,timevec] = spectrogram(signal,hann(200),150,500,srate);

subplot(4,1,2:4)
contourf(timevec,frex,abs(powspect),40,'linecolor','none')
set(gca,'ylim',[0 40])
colormap hot
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%% a simplified version to show the mechanics

% plot the signal
figure(25), clf
subplot(411)
plot(time,signal,'linew',1)
xlabel('Time (s)')
title('Time-domain signal')


n  = 500;
hz = linspace(0,srate,n+1);
tf = zeros(floor(pnts/n)-1,length(hz));
tv = zeros(floor(pnts/n)-1,1);

for i=1:floor(pnts/n)-1
    
    % cut some signal
    datasnip = signal(i*n:(i+1)*n);
    
    % compute power in this snippet
    pw = ;
    tf(i,1:length(hz)) = pw(1:length(hz));
    
    % center time point
    tv(i) = mean(time(i*n:(i+1)*n));
end

% and plot
subplot(4,1,2:4)
imagesc(tv,hz,tf')
axis xy
set(gca,'ylim',[0 40])
colormap hot
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%%
% 
%  VIDEO: Scale-free dynamics via detrended fluctuation analysis
% 
% 

clear
load dfa_data.mat

% try with narrowband amplitude time series
xfilt = abs(hilbert(filterFGx(x,srate,10,5)));

% create data with DFA=.5
N = length(x);
randnoise = randn(N,1);


% setup parameters
nScales = 20;
ranges  = round(N*[.01 .2]);
scales  = ceil(logspace(log10(),log10(),));
rmses   = zeros{};

% plot the two signals
figure(11), clf
subplot(221)
plot(timevec,randnoise)
title('Signal 1: white noise')
xlabel('Time (seconds)')

subplot(222)
plot(timevec,xfilt)
title('Signal 2: real data')
xlabel('Time (seconds)')



% integrate and mean-center the signals
randnoise = 
x4dfa = 

% and show those time series for comparison
subplot(223)
plot(timevec,randnoise)
title('Integrated noise')
 
subplot(224)
plot(timevec,x4dfa)
title('Integrated signal')

%%

% compute RMS over different time scales
for scalei = 1:nScales
    
    % number of epochs for this scale
    n = floor(N/scales(scalei)); 
    
    % compute RMS for the random noise
    epochs  = reshape( randnoise(1:n*scales(scalei)) ,scales(scalei),n);
    depochs = detrend(epochs);
    % here is the root mean square computation
    rmses(1,scalei) = mean( sqrt( mean(depochs.^2,2) ) );
    
    % repeat for the signal
    
end


% fit a linear model to quantify scaling exponent
A = [ ones(nScales,1) log10(scales)' ];  % linear model
dfa1 = (A'*A) \ (A'*log10(rmses(1,:))'); % fit to noise
dfa2 = (A'*A) \ (A'*log10(rmses(2,:))'); % fit to signal


% plot the 'linear' fit (in log-log space)
figure(12), clf, hold on

% plot results for white noise
plot(log10(scales),log10(rmses(1,:)),'rs','linew',2,'markerfacecolor','w','markersize',10)
plot(log10(scales),dfa1(1)+dfa1(2)*log10(scales),'r--','linew',2)

% plot results for the real signal
plot(log10(scales),log10(rmses(2,:)),'bs','linew',2,'markerfacecolor','w','markersize',10)
plot(log10(scales),dfa2(1)+dfa2(2)*log10(scales),'b--','linew',2)

legend({'Data (noise)';[ 'Fit (DFA=' num2str(round(dfa1(2),3)) ')' ]; ...
        'Data (signal)';[ 'Fit (DFA=' num2str(round(dfa2(2),3)) ')' ] })
xlabel('Data scale (log)'), ylabel('RMS (log)')
title('Comparison of Hurst exponent for different noises')
axis square

%% DFA scanning through frequencies

% parameters and initializations
frex = linspace(1,40,80);
dfas = zeros(size());
rms  = zeros(1,nScales);

for fi=1:length(frex)
    
    % get power time series
    x4dfa = abs(hilbert(filterFGx(x,srate,frex(fi),5)));
    
    % integrate the mean-centered signal
    x4dfa = 
    
    % compute RMS over different time scales
    for scalei = 1:nScales
        
        % number of epochs for this scale
        n = floor(N/scales(scalei));
        
        % compute RMS for this scale
        epochs  = reshape( x4dfa(1:n*scales(scalei)) ,scales(scalei),n);
        depochs = detrend(epochs);
        rms(scalei) = mean( sqrt( mean(depochs.^2,2) ) );
    end
    
    dfa = (A'*A) \ (A'*log10(rms)');
    dfas(fi) = dfa(2);
end

figure(13), clf
plot(frex,dfas,'ko-','markerfacecolor','w')
xlabel('Frequency (Hz)')
ylabel('Hurst exponent')



%% 
% 
%   VIDEO: Within-subject, cross-trial regression
% 
% 

%%% note about this dataset:
%   EEGdata is frequencies X trials
%   The goal is to test whether EEG frequency power is 
%     related to RT over trials.

% load data
load EEG_RT_data.mat
N = length(rts);

% show the data
figure(15), clf
subplot(211)
plot(,'ks-','markersize',14,'markerfacecolor','k')
xlabel('Trial'), ylabel('Response time (ms)')

subplot(212)
imagesc(1:N,frex,)
axis xy
xlabel('Trial'), ylabel('Frequency')
set(gca,'clim',[0 10])

%%% Question: Is there "a lot" or "a little" variability in RT or brain
%             over trials?

%% compute effect over frequencies



b = zeros(size(frex));

for fi=1:length(frex)
    
    % design matrix
    X = [ ones(N,1) EEGdata(fi,:)' ];
    t = (X'*X)\(X'*rts');
    
    % compute parameters, scaled by standard deviation for comparison
    % across frequencies (alternatively, you could zscore the EEG data)
    b(fi) = t(2) * std(EEGdata(fi,:));
end

% plot
figure(16), clf
subplot(211)
plot(frex,b,'rs-','markersize',14,'markerfacecolor','k')
xlabel('Frequency (Hz)')
ylabel('\beta-coefficient')


% scatterplots at these frequencies
frex2plot = dsearchn(frex',[ 8 20 ]');

for fi=1:2
    subplot(2,2,2+fi)
    
    plot(EEGdata(,:),rts,'rs','markerfacecolor','k')
    h=lsline;
    set(h,'linew',2,'color','k')
    
    xlabel('EEG energy'), ylabel('RT')
    title([ 'EEG signal at ' num2str(round(frex(frex2plot(fi)))) ' Hz' ])
end


%%% Question: How would you interpret these results?
% 


%% load EEG data and extract reaction times in ms

clear
load sampleEEGdata.mat


%%% note about the code in this cell:
%   this code extracts the reaction time from each trial
%   in the EEGLAB data format. You don't need to worry about
%   understanding this code if you do not use EEGLAB.

rts = zeros(size(EEG.epoch));

% loop over trials
for ei=1:EEG.trials
    
    % find the index corresponding to time=0, i.e., trial onset
    [~,zeroloc] = min(abs( cell2mat(EEG.epoch(ei).eventlatency) ));
    
    % reaction time is the event after the trial onset
    rts(ei) = EEG.epoch(ei).eventlatency{zeroloc+1};
end


% always good to inspect data, check for outliers, etc.
figure(17), clf
plot(rts,'ks-','markerfacecolor','w','markersize',12)
xlabel('Trial'), ylabel('Reaction time (ms)')

%% Create the design matrix
%  Our design matrix will have two regressors (two columns): intercept and RTs

X = [  ];

%% Run wavelet convolution for time-frequency analysis
%  We didn't cover this in class, but this code extracts a time-frequency
%  map of power for each trial. These power values become the dependent
%  variables.

freqrange  = [2 25]; % extract only these frequencies (in Hz)
numfrex    = 30;     % number of frequencies between lowest and highest


% set up convolution parameters
wavtime = -2:1/EEG.srate:2;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nData   = EEG.pnts*EEG.trials;
nKern   = length(wavtime);
nConv   = nData + nKern - 1;
halfwav = (length(wavtime)-1)/2;
nCyc    = logspace(log10(4),log10(12),numfrex);

% initialize time-frequency output matrix
tf3d = zeros(numfrex,EEG.pnts,EEG.trials);

% compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = fft( reshape(EEG.data(47,:,:),1,[]) ,nConv);

% loop over frequencies
for fi=1:numfrex
    
    %%% create the wavelet
    s    = nCyc(fi) / (2*pi*frex(fi));
    cmw  = exp(2*1i*pi*frex(fi).*wavtime) .* exp( (-wavtime.^2) / (2*s.^2) );
    cmwX = fft(cmw,nConv);
    cmwX = cmwX ./ max(cmwX);
    
    % second and third steps of convolution
    as = ifft( eegX .* cmwX );
    
    % cut wavelet back to size of data
    as = as(halfwav+1:end-halfwav);
    as = reshape(as,EEG.pnts,EEG.trials);
    
    % extract power from all trials
    tf3d(fi,:,:) = abs(as).^2;
    
end % end frequency loop

%% inspect the TF plots a bit

figure(18), clf

% show the raw power maps for three trials
for i=1:3
    subplot(2,3,i)
    imagesc(EEG.times,frex,squeeze(tf3d(:,:,i)))
    axis square, axis xy
    set(gca,'clim',[0 10],'xlim',[-200 1200])
    xlabel('Time (ms)'), ylabel('Frequency')
    title([ 'Trial ' num2str(i) ])
end


% now show the trial-average power map
subplot(212)
imagesc(EEG.times,frex,squeeze(mean(tf3d,3)))
axis square, axis xy
set(gca,'clim',[0 5],'xlim',[-200 1200])
xlabel('Time (ms)'), ylabel('Frequency')
title('All trials')

%% now for the regression model

% We're going to take a short-cut here, and reshape the 3D matrix to 2D.
% That doesn't change the values, and we don't alter the trial order.
% Note the size of the matrix below.
tf2d = reshape(tf3d,numfrex*EEG.pnts,EEG.trials)';


% Now we can fit the model on the 2D matrix
b = (X'*X)\X'*tf2d;

% reshape b into a time-by-frequency matrix
betamat = ;

%% show the design and data matrices

figure(19), clf

ax1_h = axes;
set(ax1_h,'Position',[.05 .1 .1 .8])
imagesc(X)
set(ax1_h,'xtick',1:2,'xticklabel',{'Int';'RTs'},'ydir','norm')
ylabel('Trials')
title('Design matrix')


ax2_h = axes;
set(ax2_h,'Position',[.25 .1 .7 .8])
imagesc(tf2d)
set(ax2_h,'ydir','norm','clim',[0 20])
ylabel('Trials')
xlabel('Timefrequency')
title('Data matrix')

colormap gray


%%% QUESTION: Please interpret the matrices! 
%             What do they mean and what do they show?
% 
% 

%% show the results

figure(20), clf

% show time-frequency map of regressors
contourf(EEG.times,frex,betamat,40,'linecolor','none')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
set(gca,'xlim',[-200 1200],'clim',[-.012 .012])
title('Regression against RT over trials')

%%% QUESTION: How do you interpret the results?
% 
% 
%%% QUESTION: Do you believe these results? Are they statistically
%             significant? 
% 
% 


%% 
% 
%   VIDEO: Downsampling time-frequency results
% 
% 

% Take one channel from the v1 dataset, all trials and all time points.
% Extract time-frequency power from 10 to 100 Hz in 42 steps and average over trials.
% Save the full temporal resolution map, and also a version temporally
%    downsampled to 40 Hz (one time point each 25 ms [approximate])
% Show images of both maps in the same figure.
% 

clear
load v1_laminar

% these are the time points we want to save
times2save = 

% now we need to find those indices in the time vector


% soft-coded parameters
freqrange  = ; % extract only these frequencies (in Hz)
numfrex    = ; % number of frequencies between lowest and highest


% set up convolution parameters
wavtime = -1:1/srate:1-1/srate;
frex    = linspace(freqrange(1),freqrange(2),numfrex);
nData   = 
nKern   = length(wavtime);
nConv   = nData^2 + 3*nKern; % hmm...
halfwav = (length(wavtime)-1)/2;


% create wavelets (do you need to re-create the wavelets? why or why not?)
cmwX = zeros(numfrex,nConv);
for fi=1:numfrex
    
    % create time-domain complex Morlet wavelet
    cmw = 
    
    % compute fourier coefficients of wavelet and normalize
    cmwX(fi,:) = fft(,);
    cmwX(fi,:) = cmwX(fi,:) ./ min(cmwX(fi,:)); % :(
end



% initialize time-frequency output matrix
tf = cell(2,1);

% compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = fft( reshape(csd(7,:,:),1,[]) ,nConv);

% loop over frequencies
for fi=1:numfrex
    
    % second and third steps of convolution
    as = 
    
    % cut wavelet back to size of data
    as = as(halfwav+1:end-halfwav);
    % reshape to time X trials
    as = reshape(
    
    % compute power from all time points
    tf{1}(fi,:) = mean(abs(as).^2,2);
    
    % compute power only from the downsampled time points
    tf{2}(fi,:) = 
    
end % end frequency loop



% visualization
figure(18), clf
titles = {'Full';'Downsampled'};

for i=1:2
    subplot(1,2,i)
    
    % select which time vector to use
    if i==1, t=timevec; else, t=times2save; end
    
    contourf(t,frex,tf{i},40,'linecolor','none');
    set(gca,'clim',[0 10000],'xlim',[-.2 1.2])
    xlabel('Time (s)'), ylabel('Frequencies (Hz)')
    title(
end


%%% QUESTION: Repeat with more downsampling. How low can the new sampling rate be
%              before the data become difficult to interpret?




%% 
% 
%   VIDEO: Linear vs. logarithmic frequency scaling
% 
% 


% clear the workspace and load in the data
clear
load v1_laminar.mat


% soft-coded parameters
freqrange  = [10 100]; % extract only these frequencies (in Hz)
numfrex    = 1;        % number of frequencies between lowest and highest

% select a log or linear frequency scaling
logOrLin = 'lin';



% select frequency range
if logOrLin(2)=='o'
    frex = logspace(log10(freqrange(1)),log10(freqrange(2)),numfrex);
else
    frex = linspace(freqrange(1),freqrange(2),numfrex);
end

% set up convolution parameters
wavtime = -1:1/srate:1-1/srate;
nData   = length(timevec)*size(csd,3);
nKern   = length(wavtime);
nConv   = 
halfwav = (length(wavtime)-1)/2;


% create wavelets
cmwX = zeros(numfrex,nConv);
for fi=1:numfrex
    
    % create time-domain wavelet
    cmw = sqrt(hello) .* exp( -4*log(2)*wavtime.^2 / .3.^2 );
    
    % compute fourier coefficients of wavelet and normalize
    cmwX(fi,:) = fft(cmw,nConv);
    cmwX(fi,:) = cmwX(fi,:) ./ max(cmwX(fi,:));
end



% initialize time-frequency output matrix
tf = zeros(numfrex,pnts);

% compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = fft( reshape(csd(7,:,:),1,[]) ,nConv);

% loop over frequencies
for fi=1:numfrex
    
    % second and third steps of convolution
    as = ifft( cmwX(fi,:).*eegX ,.5 );
    
    % cut wavelet back to size of data
    as = as(halfwav+1:end-halfwav);
    % reshape to time X trials
    as = reshape(as,size(csd,2),size(csd,3));
    
    % compute power from all time points
    tf(fi,:) = abs(as).^2;
    
end % end frequency loop



% visualization in separate figures
if logOrLin(2)=='o'
    figure(16+1), clf
else
    figure(16+2), clf
end

contourf(timevec,frex,tf,40,'linecolor','none');
set(gca,'clim',[0 10000],'xlim',[-.2 1.2])
xlabel('Time (s)'), ylabel('Frequencies (Hz)')
title([ logOrLin ' frequency scalng' ])

%%% QUESTION: Is there a difference in the y-axis scaling?
% 

%% Manually change the y-axis scaling to log

set(gca,'YScale','')
set(gca,'ytick',round(frex(1:5:numfrex),2))


%% 
% 
%   VIDEO: Total, non-phase-locked, and phase-locked power
% 
% 

%    Here you will separate the non-phase-locked component of the signal
%      to see if they differ. Use the scalp EEG data from channel PO7.


clear % a clear MATLAB workspace is a clear mental workspace

% start by loading in the data and picking a channel to work with
load sampleEEGdata.mat
chan2use = 'po7';

% find the index in the data with that channel label
chanidx = strcmpi(chan2use,{EEG.chanlocs.labels});

%%% How to separate the non-phase-locked signal:
%  Start from the assumption that the ERP reflects the phase-locked
%   component. This means that the single-trial data contains the
%   phase-locked PLUS the non-phase-locked activity.
%  Therefore, the non-phase-locked component is obtained by 
%   subtracting the ERP from the single-trial data.

% But, you need to keep both, so you'll need a separate variable for the
% non-phase-locked signal.
EEG.NPL = EEG.data; % NPL = non-phase-locked

% I'm going to do this over all channels, although the instructions are
% really only for one channel.
for chani=1:EEG.nbchan
    % compute ERP from this channel
    thisChanERP = 
    
    % use bsxfun to subtract that ERP from all trials
    EEG.NPL(chani,:,:) = bsxfun(@minus,EEG.data(chani,:,:),thisChanERP);
end

% note: can also be done without a loop and without initializing
EEG.NPL = 

%%%% NOTE: If you are doing this in real data, the non-phase-locked part of
%%%% the signal should be computed separately for each condition. This
%%%% avoids artificial contamination by ERP condition differences.

%% Plot the ERP from channel poz for the total and non-phase-locked signal

figure(19), clf, hold on
plot(times,mean(EEG.data(chanidx,:,:),3),'b','linew',2)
plot(times,mean(EEG.NPL(chanidx,:,:),3),'r','linew',2)
xlabel('Time (ms)'), ylabel('Activity (\muV)')
title('Data.')
set(gca,'xlim',[-300 1200])
legend({'Total';'Non-phase-locked'})

%%% QUESTION: Are you surprised at the red line? Is it a feature or a bug?!!?
% 


%% now for time-frequency analysis
%  apply wavelet convolution to both signal components.
%  extract trial-averaged power and ITPC.
%  You can pick the parameters; a reasonable frequency range is 2-40 Hz


% wavelet parameters
num_frex = ;
min_freq = ;
max_freq = ;

% parameters
frex  = linspace(min_freq,max_freq,num_frex);
time  = -1:1/EEG.srate:1;
half_wave = (length(time)-1)/2;

% FFT parameters
nWave = length(time);
nData = EEG.pnts*EEG.trials;
nConv = nWave+nData-1;

% baseline time window
baseidx = dsearchn(EEG.times',[-500 -200]');


% FFT of data (all time points, all trials)
dataXTotal = fft( reshape(EEG.data(chanidx,:,:),1,nData) ,nConv);
dataXNPL   = fft( reshape(EEG.NPL(chanidx,:,:),1,nData) ,nConv);


% initialize output time-frequency data
% Note: the first '2' is for total/NPL; the second '2' is for power/ITPC
tf = zeros(2,num_frex,EEG.pnts,2);

% loop over frequencies
for fi=1:num_frex
    
    % create wavelet and get its FFT
    fwhm = .3;
    wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp( -4*log(2)*time.^2/fwhm^2 );
    waveletX = fft(wavelet,nConv);
    
    %%% convolution on total activity
    as = ifft(waveletX.*dataXTotal,nConv);
    as = as(half_wave+1:end-half_wave);
    as = reshape(as,EEG.pnts,EEG.trials);
    
    % compute power and ITPC
    basepow = mean(mean(abs(as(baseidx(1):baseidx(2),:).^2),1),2);
    tf(1,fi,:,1) = 10*log10( mean(abs(as.^2),2) / basepow );
    tf(1,fi,:,2) = abs(mean(exp(1i*angle(as)),2));
    
    
    
    %%% repeat for non-phase-locked activity
    as = ifft(waveletX.*dataXNPL,nConv);
    as = as(half_wave+1:end-half_wave);
    as = reshape(as,EEG.pnts,EEG.trials);
    
    % compute power and ITPC
    basepow = mean(mean(abs(as(baseidx(1):baseidx(2),:).^2),1),2);
    tf(2,fi,:,1) = 10*log10( mean(abs(as.^2),2) / basepow );
    tf(2,fi,:,2) = abs(mean(exp(1i*angle(as)),2));
end

%% Now for plotting. 
%  Make a 2x3 grid of imagesc, showing power (top row) and ITPC (bottom row)
%   for total (left column) and non-phase-locked (middle column) activity

figure(20), clf

% total power
subplot(231)
contourf(EEG.times,frex,squeeze(tf(1,:,:,1)),40,'linecolor','none')
set(gca,'xlim',[-300 1200],'clim',[-1 1]*3)
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Total power')

% non-phase-locked power
subplot(232)
contourf(EEG.times,frex,squeeze(tf(2,:,:,1)),40,'linecolor','none')
set(gca,'xlim',[-300 1200],'clim',[-1 1]*3)
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Non-phase-locked power')


% total ITPC
subplot(234)
contourf(EEG.times,frex,squeeze(tf(1,:,:,2)),40,'linecolor','none')
set(gca,'xlim',[-300 1200],'clim',[0 .5])
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Total ITPC')

% non-phase-locked ITPC
subplot(235)
contourf(EEG.times,frex,squeeze(tf(2,:,:,2)),40,'linecolor','none')
set(gca,'xlim',[-300 1200],'clim',[0 .5])
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Non-phase-locked ITPC')


%%% QUESTION: Are you surprised at the non-phase-locked ITPC result?
% 
% 
%%% QUESTION: What features are the same vs. different between the power plots?
% 


%% Now for the phase-locked power.


% Some people compute the phase-locked power as the time-frequency analysis
%  of the ERP. Although theoretically sensible, this can be unstable,
%  particularly if a baseline normalization is applied.
% A more stable method is to take the difference between total and
%  non-phase-locked power.

subplot(233)
contourf(EEG.times,frex,squeeze(tf(1,:,:,1)-tf(2,:,:,1)),40,'linecolor','none')
set(gca,'xlim',[-300 1200],'clim',[-1 1]*3)
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Phase-locked power')


%%% QUESTION: How does the phase-locked power compare to the total ITPC?
% 
% 
%%% QUESTION: Does it make sense to compute the phase-locked ITPC? Why?
% 


%% done.
