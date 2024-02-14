%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Multivariate components analysis
%      VIDEO: Frequency-based GED for source-separation in simulated data
% Instructor: sincxpress.com
%
%%

% generate the data
NTSA_multivar_simData
close

% always a good idea to inspect the data...
EEG

%% create S and R covariance matrices based on spectra

% filter the data in broad alpha
EEG.fdata = filterFGx(EEG.data,EEG.srate,11,7);

% wide time window
tidx = dsearchn(EEG.times',[ 0 2 ]');

% initialize covariance matrices
[S,R] = deal( zeros(EEG.nbchan) );

% loop over trials and create covariance matrices
for triali=1:EEG.trials
    
    %%% R matrix
    tmp = EEG.data(:,tidx(1):tidx(2),triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    R   = R + tmp*tmp'/diff(tidx);
    
    %%% R matrix
    tmp = EEG.fdata(:,tidx(1):tidx(2),triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    S   = S + tmp*tmp'/diff(tidx);
end

% divide by N to finish averaging
S = S/EEG.trials;
R = R/EEG.trials;


% set dynamic color limits
clim = [-1 1]*max(abs([S(:); R(:)]))*.7;

% let's have a look
figure(1), clf
subplot(121)
imagesc(S), axis square
set(gca,'clim',clim)
xlabel('Channel'), ylabel('Channel')
title('S covariance matrix')


subplot(122)
imagesc(R), axis square
set(gca,'clim',clim)
xlabel('Channel'), ylabel('Channel')
title('R covariance matrix')

%% now for GED

% GED and sort
[V,D] = eig(S,R);
[d,sidx] = sort(diag(D),'descend');
V = V(:,sidx);

figure(2), clf
plot(d,'ks-','markerfacecolor','w','linew',2,'markersize',13)
xlabel('Component #'), ylabel('Power ratio (\lambda)')
title('Scree plot')


% compute component time series and add as extra channels
cmpdat = V(:,1:2)'*reshape(EEG.data(1:EEG.nbchan,:,:),EEG.nbchan,[]);
EEG.data(EEG.nbchan+1:EEG.nbchan+2,:,:) = reshape( cmpdat,[2 EEG.pnts EEG.trials] );



% plot the two components
plot_simEEG(EEG,65,3)
subplot(221)
topoplotIndie(V(:,1)'*S,EEG.chanlocs);
title('Component 1')
subplot(222)
topoplotIndie(-lf.Gain(:,1,diploc1), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Ground truth')

plot_simEEG(EEG,66,4)
subplot(221)
topoplotIndie(V(:,2)'*S,EEG.chanlocs);
title('Component 2')
subplot(222)
topoplotIndie(-lf.Gain(:,1,diploc2), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Ground truth')

%% done.
