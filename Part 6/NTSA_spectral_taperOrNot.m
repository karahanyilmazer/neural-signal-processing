%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: To taper or not to taper?
% Instructor: sincxpress.com
%
%%

% parameters
N  = 200; % make it even!
tv = -N/2:N/2;
tv = tv / std(tv);

% create tapers
hanntaper = .5*(1-cos(2*pi*(0:N)/N));
gaustaper = exp(-tv.^2);
hammtaper = .54 - .46*cos(2*pi*(0:N)/N);

% plot the tapers
figure(1), clf, hold on
plot(tv,hanntaper,'linew',2)
plot(tv,hammtaper,'linew',2)
plot(tv,gaustaper,'linew',2)
legend({'Hann';'Gauss';'Hamming'})
xlabel('Time (a.u.)'), ylabel('Gain')

% tapers differentiate more in logspace
%set(gca,'YScale','log')

%% data before and after being tapered

% create data as random numbers
data = randn(1,N+1);

% taper the data
dataHann = data.*hanntaper;
dataGaus = data.*gaustaper;
dataHamm = data.*hammtaper;


% various plotting
figure(2), clf, hold on
subplot(311)
plot(tv,data, tv,dataHann,'linew',2)
title('With Hann taper')
set(gca,'xlim',tv([1 end]))

subplot(312)
plot(tv,data, tv,dataHamm,'linew',2)
title('With Hamming taper')
set(gca,'xlim',tv([1 end]))

subplot(313)
plot(tv,data, tv,dataGaus,'linew',2)
title('With Gaussian taper')
set(gca,'xlim',tv([1 end]))

xlabel('Time (a.u.)'), ylabel('Activity')

%% Power spectra

% compute power
powData = abs(fft(data)).^2;
powHann = abs(fft(dataHann)).^2;
powHamm = abs(fft(dataHamm)).^2;
powGaus = abs(fft(dataGaus)).^2;

% frequencies vector
hz = linspace(0,1,N+1);

% plots
figure(3), clf, hold on
h(1) = plot(hz,powData,'rs-');
h(2) = plot(hz,powHann,'bo-');
h(3) = plot(hz,powHamm,'k^-');
h(4) = plot(hz,powGaus,'mp-');

% adjust the line properties
set(h,'linew',2,'markersize',10,'markerfacecolor','w')

% make the plot look a bit nicer
legend({'None';'Hann';'Hamming';'Gauss'})
set(gca,'ytick',[],'xlim',[0 .1])
xlabel('Frequency (norm.)'), ylabel('Power (a.u.)')

%%

%% another example signal

% new signal with nonstationarity
data = sin(linspace(0,20*pi,N+1));
data = data + linspace(-1,2,N+1);


% taper again
dataHann = data.*hanntaper;
dataGaus = data.*gaustaper;
dataHamm = data.*hammtaper;


% various plotting
figure(4), clf, hold on
subplot(311)
plot(tv,data, tv,dataHann,'linew',2)
title('With Hann taper')
set(gca,'xlim',tv([1 end]))

subplot(312)
plot(tv,data, tv,dataHamm,'linew',2)
title('With Hamming taper')
set(gca,'xlim',tv([1 end]))

subplot(313)
plot(tv,data, tv,dataGaus,'linew',2)
title('With Gaussian taper')
set(gca,'xlim',tv([1 end]))

xlabel('Time (a.u.)'), ylabel('Activity')


%% power spectra again

% compute power
powData = abs(fft(data)).^2;
powHann = abs(fft(dataHann)).^2;
powHamm = abs(fft(dataHamm)).^2;
powGaus = abs(fft(dataGaus)).^2;

% frequencies vector
hz = linspace(0,1,N+1);

% plots
figure(5), clf, hold on
h(1) = plot(hz,powData,'rs-');
h(2) = plot(hz,powHann,'bo-');
h(3) = plot(hz,powHamm,'k^-');
h(4) = plot(hz,powGaus,'mp-');

% adjust the line properties
set(h,'linew',2,'markersize',10,'markerfacecolor','w')

% make the plot look a bit nicer
legend({'None';'Hann';'Hamming';'Gauss'})
set(gca,'ytick',[],'xlim',[0 .1])
xlabel('Frequency (norm.)'), ylabel('Power (a.u.)')

%%

%% now with real data

% import data
load EEGrestingState.mat

% extract epochs and taper each epoch
N = 2048;
epochs = reshape(eegdata,N,[]);
epochs = cat(2,epochs,reshape(eegdata(N/2+1:end-N/2),N,[]));
epochsTap = bsxfun(@times,epochs,.5*(1-cos(2*pi*(0:N-1)/(N-1)))');


% FFT of tapered and untapered data
fNot = mean( abs(fft( epochs    )/N).^2 ,2);
fHan = mean( abs(fft( epochsTap )/N).^2 ,2);

% frequencies vector
hz = linspace(0,srate/2,floor(N/2)+1);


% plot on example epoch
figure(6), clf
subplot(211), hold on
plot(epochs(:,1),'k')
plot(epochsTap(:,1),'r')
% and the next overlapping epoch
plot(N/2:N/2+N-1,epochs(:,61)-40,'b')
plot(N/2:N/2+N-1,epochsTap(:,61)-40,'m')
% plotting niceties
set(gca,'xlim',[0 3*N/2],'ytick',[],'xtick',[])
legend({'e_1: None';'e_1: Hann';'e_2: None';'e_2: Hann'})


% now plot power spectra
subplot(212), hold on
plot(hz,fNot(1:length(hz)),'k','linew',2)
plot(hz,fHan(1:length(hz)),'r','linew',2)
xlabel('Frequency (Hz)')
set(gca,'xlim',[0 60])
legend({'None';'Hann'})


%% done.
