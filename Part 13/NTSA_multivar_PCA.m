%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Multivariate components analysis
%      VIDEO: Principal components analysis (PCA) of simulated data
% Instructor: sincxpress.com
%
%%

% generate the data
NTSA_multivar_simData

% always a good idea to inspect the data...
EEG

%% single-trial covariance

% time window
tidx = dsearchn(EEG.times',[0 800]');

covmatT = zeros( EEG.nbchan );
for triali=1:EEG.trials
    tmpdat  = EEG.data(:,tidx(1):tidx(2),triali);
    tmpdat  = bsxfun(@minus,tmpdat,mean(tmpdat,2));
    covmatT = covmatT + tmpdat*tmpdat' / diff(tidx);
end

% divide by N to finish the averaging
covmatT = covmatT / EEG.trials;

%% PCA decomposition

% eigendecomposition of covariance matrix
[evecs,evals] = eig(covmatT);

% sort according to eigenvalues
[evals,sidx] = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);

% convert evals to % change
evalsPC = 100*evals./sum(evals);


% show matrices
figure(1), clf

subplot(131), imagesc(evecs)
title('Eigenvectors'), axis square

subplot(132), imagesc(diag(log(evals)))
title('Eigenvalues'), axis square

subplot(233)
plot(evals,'ks-','markerfacecolor','w','markersize',8,'linew',2)
axis square
xlabel('Component'), ylabel('Eigenvalue (\lambda)')

subplot(236)
plot(evalsPC,'ks-','markerfacecolor','w','markersize',8,'linew',2)
axis square
xlabel('Component'), ylabel('Variance explained (%)')

%% component time series and maps

data2d = reshape(EEG.data(1:EEG.nbchan,:,:),EEG.nbchan,[]);
compTS = evecs(:,1:2)'*data2d;
EEG.data(EEG.nbchan+1:EEG.nbchan+2,:,:) = reshape(compTS,[2 EEG.pnts EEG.trials]);

% Just because you can't doesn't mean you should.
%EEG.data(EEG.nbchan+1:EEG.nbchan+2,:,:) = reshape(evecs(:,1:2)'*reshape(EEG.data,EEG.nbchan,[]),[2 EEG.pnts EEG.trials]);

% plot the two components
plot_simEEG(EEG,65,2)
subplot(221)
topoplotIndie(evecs(:,1),EEG.chanlocs);
title('Component 1')
subplot(222)
topoplotIndie(-lf.Gain(:,1,diploc1), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Ground truth')


plot_simEEG(EEG,66,3)
subplot(221)
topoplotIndie(evecs(:,2),EEG.chanlocs);
title('Component 2')
subplot(222)
topoplotIndie(-lf.Gain(:,1,diploc2), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Ground truth')

%% the MATLAB built-in way

% extract data
tmpdat = reshape( EEG.data(1:EEG.nbchan,tidx(1):tidx(2),:),EEG.nbchan,[] );

% run PCA using MATLAB function (stats toolbox)
[coeff,score,latent, tsquared, explained, mu] = pca(tmpdat);


%%% plotting for comparisons
figure(4), clf
subplot(211), hold on
plot(1:EEG.nbchan,evalsPC,'ks-','markerfacecolor','w','markersize',10)
plot((1:length(explained))+.25,explained,'ro-','markerfacecolor','w','markersize',10)
set(gca,'xlim',[0 20])
xlabel('Component'), ylabel('Variance explained (%)')
title('Eigenvalues (scree plot)')
legend({'Manual';'pca()'})

subplot(212), hold on
plot(1:EEG.nbchan,zscore(evecs(:,1)),'ks-','markerfacecolor','w','markersize',10)
plot((1:EEG.nbchan)+.25,zscore(score(:,1)),'ro-','markerfacecolor','w','markersize',10)
set(gca,'xlim',[0 20])
xlabel('Component'), ylabel('Weight (norm.)')
title('Eigenvectors')
legend({'Manual';'pca()'})

%% done.
