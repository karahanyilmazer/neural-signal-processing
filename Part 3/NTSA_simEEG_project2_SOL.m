%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Project 2 SOLUTIONS!
% Instructor: sincxpress.com
%
%%

%% 

% mat file containing EEG, leadfield and channel locations
load emptyEEG

% select dipole location (more-or-less random)
diploc = 109;

% plot brain dipoles
figure(1), clf, subplot(121)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'bo','markerfacecolor','y')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 'rs','markerfacecolor','k','markersize',10)
rotate3d on, axis square
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(122)
topoplotIndie(-lf.Gain(:,1,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
set(gca,'clim',[-1 1]*40)
title('Signal dipole projection')

%% add signal to one dipole and project to scalp

% reduce data size a bit
EEG.pnts  = 2000;
EEG.times = (0:EEG.pnts-1)/EEG.srate;

% initialize all dipole data
dipole_data = zeros(size(lf.Gain,3),EEG.pnts);

% add signal to one dipole
dipole_data(diploc,:) = sin(2*pi*10*EEG.times);

% now project dipole data to scalp electrodes
EEG.data = squeeze(lf.Gain(:,1,:))*dipole_data;

% plot the data
plot_simEEG(EEG,31,2);

%% now for the projects!

%%%% IMPORTANT! Check internal consistency with existing EEG structure!!!

EEG


%% 1) pure sine wave with amplitude explorations

EEG.trials = 40;

% dipole amplitude magnitude
ampl = 1e-30;

% initialize all dipole data
dipole_data = zeros(size(lf.Gain,3),EEG.pnts);
dipole_data(diploc,:) = ampl * sin(2*pi*10*EEG.times);

% compute one trial
signal = squeeze(lf.Gain(:,1,:))*dipole_data;

% repeat that for N trials
EEG.data = repmat(signal,[1 1 EEG.trials]);


% plot the data
plot_simEEG(EEG,31,2);

%%% Question: What is the smallest amplitude of dipole signal that still
%             elicits a scalp-level response?

%% 2) sine wave with noise

%%% Question: Given amplitude=1 of dipole signal, what standard deviation of noise
%             at all other dipoles overpowers the signal (qualitatively)?


% noise standard deviation
noiseSD = 5;

for triali=1:EEG.trials
    
    % initialize all dipole data
    dipole_data = noiseSD * randn(size(lf.Gain,3),EEG.pnts);
    dipole_data(diploc,:) = sin(2*pi*10*EEG.times);
    
    % compute one trial
    EEG.data(:,:,triali) = squeeze(lf.Gain(:,1,:))*dipole_data;
end

% plot the data
plot_simEEG(EEG,31,2);


%% 3) Non-oscillatory transient in one dipole, noise in all other dipoles

% Gaussian
peaktime = 1; % seconds
width    = .12;
ampl     = 70;

% create Gaussian taper
gaus = ampl * exp( -(EEG.times-peaktime).^2 / (2*width^2) );


for triali=1:EEG.trials
    
    % initialize all dipole data
    dipole_data = randn(size(lf.Gain,3),EEG.pnts);
    dipole_data(diploc,:) = gaus;
    
    % compute one trial
    EEG.data(:,:,triali) = squeeze(lf.Gain(:,1,:))*dipole_data;
end


plot_simEEG(EEG,31,2);

%% 4) Non-stationary oscillation in one dipole, transient oscillation in another dipole, noise in all dipoles

%%% first pick two dipoles

% select dipole location
diploc1 = 109;
diploc2 = 510;

% plot brain dipoles
figure(1), clf, subplot(131)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'bo','markerfacecolor','y')
hold on
plot3(lf.GridLoc(diploc1,1), lf.GridLoc(diploc1,2), lf.GridLoc(diploc1,3), 'ks','markerfacecolor','k','markersize',10)
plot3(lf.GridLoc(diploc2,1), lf.GridLoc(diploc2,2), lf.GridLoc(diploc2,3), 'rs','markerfacecolor','r','markersize',10)
rotate3d on, axis square
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(132)
topoplotIndie(-lf.Gain(:,1,diploc1), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
set(gca,'clim',[-1 1]*40)
title('Signal dipole projection')

subplot(133)
topoplotIndie(-lf.Gain(:,1,diploc2), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
set(gca,'clim',[-1 1]*40)
title('Signal dipole projection')

%% then do the simulation

% Gaussian and sine parameters
peaktime = 1; % seconds
width    = .12;
sinefreq = 7; % for sine wave

% create Gaussian taper
gaus = exp( -(EEG.times-peaktime).^2 / (2*width^2) );


% initialize EEG.data
EEG.data = zeros(EEG.nbchan,EEG.pnts,EEG.trials);

for triali=1:EEG.trials
    
    % initialize all dipole data
    dipole_data = randn(size(lf.Gain,3),EEG.pnts)/5;
    
    % non-stationary oscillation in dipole1
    % range of 5-10 Hz
    freqmod = 5 + 5*interp1(rand(1,10),linspace(1,10,EEG.pnts));
    dipole_data(diploc1,:) = sin( 2*pi * ((EEG.times + cumsum(freqmod))/EEG.srate) );
    
    
    % transient oscillation in dipole2
    dipole_data(diploc2,:) = sin( 2*pi*sinefreq*EEG.times + rand*pi ) .* gaus;
    
    
    % compute one trial of data
    EEG.data(:,:,triali) = squeeze(lf.Gain(:,1,:))*dipole_data;
end

% plot the data
plot_simEEG(EEG,56,3);
plot_simEEG(EEG,31,2);

%% 
