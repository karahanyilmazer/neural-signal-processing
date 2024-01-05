%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Spectral analyses
%      VIDEO: Spectral separation on simulated dipole data
% Instructor: sincxpress.com
%
%%

load emptyEEG

% select dipole location
diploc1 = 109;
diploc2 = 118;

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

%% adjust EEG parameters

EEG.pnts   = 1143;
EEG.trials = 150;
EEG.times  = (0:EEG.pnts-1)/EEG.srate - .2;

% initialize EEG data
EEG.data = zeros([ EEG.nbchan EEG.pnts EEG.trials ]);

%% create simulated data

% Gaussian
peaktime = .5; % seconds
fwhm     = .12;

% create Gaussian taper
gaus = exp( -(4*log(2)*(EEG.times-peaktime).^2) / fwhm^2 );

sinefreq1 = 9;
sinefreq2 = 14;

sine1 = sin(2*pi*sinefreq1*EEG.times);
sine2 = sin(2*pi*sinefreq2*EEG.times);

for triali=1:EEG.trials
    
    % initialize all dipole data
    dipdat = .01 * randn(size(lf.Gain,3),EEG.pnts);
    
    dipdat(diploc1,:) = sine1 .* gaus;
    dipdat(diploc2,:) = sine2 .* gaus;
    
    % compute one trial
    EEG.data(:,:,triali) = squeeze(lf.Gain(:,1,:))*dipdat;
end

% try a few channels...
plot_simEEG(EEG,20,2);
plot_simEEG(EEG,30,3);
plot_simEEG(EEG,29,4);

%% done.
