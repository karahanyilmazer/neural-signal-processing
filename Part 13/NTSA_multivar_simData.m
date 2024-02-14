%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Multivariate components analysis
%      VIDEO: Simulate multicomponent EEG data
% Instructor: sincxpress.com
%
%%

%% 

% mat file containing EEG, leadfield and channel locations
load emptyEEG

% pick two dipoles
diploc1 = 109;
diploc2 = 409;

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

%% adjust parameters

EEG.times  = (-EEG.srate:2*EEG.srate)/EEG.srate;
EEG.pnts   = numel(EEG.times);
EEG.trials = 100;

% initialize channel data
EEG.data = zeros(EEG.nbchan,EEG.pnts,EEG.trials);

%% generate dipole -> EEG activity

% fixed IF signal for dipole1
gaus1 = exp( -4*log(2)*(EEG.times-0.8).^2 );
gaus2 = exp( -4*log(2)*(EEG.times-0.9).^2 / .8 );

%%% loop over channels
for triali=1:EEG.trials
    
    % begin with random dipole activity
    dipact = .02 * randn(size(lf.Gain,3),EEG.pnts);
    
    
    %%% dipole 1
    freqmod  = 5+5*interp1(rand(1,10),linspace(1,10,EEG.pnts));
    IFsignal = sin( 2*pi * ((EEG.times + cumsum(freqmod))/EEG.srate) );
    dipact(diploc1,:) = IFsignal .* gaus1;
    
    
    %%% dipole 2
    freqmod  = 7+5*interp1(rand(1,10),linspace(1,10,EEG.pnts));
    IFsignal = sin( 2*pi * ((EEG.times + cumsum(freqmod))/EEG.srate) );
    dipact(diploc2,:) = IFsignal .* gaus2;
    
    % now project to the scalp
    EEG.data(:,:,triali) = squeeze(lf.Gain(:,1,:))*dipact;
end


% plot data from a few channels
if 0 % set to 0 for simulation-only
    plot_simEEG(EEG,31,2)
    plot_simEEG(EEG,18,3)
    plot_simEEG(EEG,10,4)
end

%% done.
