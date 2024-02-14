%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Multivariate components analysis
%      VIDEO: Project 6-1 SOLUTIONS: GED for interacting alpha sources
% Instructor: sincxpress.com
%
%%

load('restingstate.mat')

% inspect the data a bit...
EEG

plot_simEEG(EEG,31,10)

%% source separation for alpha

% filter the data in broad alpha
EEG.fdata = filterFGx(EEG.data,EEG.srate,11,7);

% initialize covariance matrices
[S,R] = deal( zeros(EEG.nbchan) );

% loop over trials and create covariance matrices
for triali=1:EEG.trials
    
    %%% R matrix
    tmp = EEG.data(:,:,triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    R   = R + tmp*tmp'/EEG.pnts;
    
    %%% R matrix
    tmp = EEG.fdata(:,:,triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    S   = S + tmp*tmp'/EEG.pnts;
end

% divide by N to finish averaging
S = S/EEG.trials;
R = R/EEG.trials;


% set dynamic color limits
clim = [-1 1]*max(abs([S(:); R(:)]))*.2;

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
subplot(211)
topoplotIndie(V(:,1)'*S,EEG.chanlocs);
title('Component 1')

plot_simEEG(EEG,66,4)
subplot(211)
topoplotIndie(V(:,2)'*S,EEG.chanlocs);
title('Component 2')

%% phase synchronization between the top two components over frequencies

% list frequencies and filter FWHM
frex = linspace(2,50,100);
fwhm = linspace(1,5,length(frex));

% initialize
compsynch = zeros(size(frex));
comppower = zeros(length(frex),2);

% loop over frequencies
for fi=1:length(frex)
    
    % narrowband filter and extract analytic envelop
    fdat = filterFGx(EEG.data(65:66,:,:),EEG.srate,frex(fi),fwhm(fi));
    as   = hilbert( reshape(fdat,2,[])' ).';
    
    % compute synchronization and average power
    compsynch(fi)   = abs(mean( exp(1i*diff(angle(as),[],1)) ));
    comppower(fi,:) = mean(abs(as).^2,2);
end


% plot phase synchronization spectrum
figure(5), clf
subplot(211)
plot(EEG.times,angle(as(:,1:EEG.pnts)))
xlabel('Time (ms)'), ylabel('Phase angle (rad.)')
title('Example phase angle time series')

subplot(212)
plot(frex,compsynch,'k-s','linew',2,'markersize',10,'markerfacecolor','w')
xlabel('Freuqency (Hz)'), ylabel('Synchronization')
title('Phase synchronization spectrum')



% plot power spectrum
figure(6), clf
subplot(211)
plot(EEG.times,abs(as(:,1:EEG.pnts)).^2)
xlabel('Time (ms)'), ylabel('Power (\muV^2)')
title('Example amplitude time series')

subplot(212)
plot(frex,comppower,'-s','linew',2,'markersize',10,'markerfacecolor','w')
xlabel('Frequency (Hz)'), ylabel('Power')
title('Power spectrum')


%% amplitude time series correlations across the top 8 components

% get component time series
top8 = V(:,1:8)'*reshape(EEG.data(1:EEG.nbchan,:,:),EEG.nbchan,[]);

% extract amplitude time series
top8 = abs(hilbert( filterFGx(top8,EEG.srate,11,7)' ).');

% correlation matrix
cormat = corrcoef(top8');

figure(7), clf
subplot(211)
plot(EEG.times,top8(:,1:EEG.pnts))
xlabel('Time (ms)'), ylabel('Power')
title('Example power time series')

subplot(212)
imagesc(cormat), axis square
xlabel('Component'), ylabel('Component')
title('Correlation matrix')
set(gca,'clim',[-1 1]*.7)
colorbar


% finally, show all topoplots
figure(8), clf
for i=1:8
    subplot(2,4,i)
    topoplotIndie(V(:,i)'*S,EEG.chanlocs,'numcontour',0,'plotrad',.65);
    title([ 'Component ' num2str(i) ])
end

%% done.
