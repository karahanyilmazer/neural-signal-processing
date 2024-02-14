%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Multivariate components analysis
%      VIDEO: Create covariance matrices based on time and on frequency
% Instructor: sincxpress.com
%
%%

% generate the data
NTSA_multivar_simData

% always a good idea to inspect the data...
EEG

%% average covariance from single-trials

covmatT = zeros( EEG.nbchan );

for triali=1:EEG.trials
    
    % the "manual" way (isolate, center, transpose)
    tmpdat  = EEG.data(:,:,triali);
    tmpdat  = bsxfun(@minus,tmpdat,mean(tmpdat,2));
    covmatT = covmatT + tmpdat*tmpdat' / EEG.pnts;
    
    % the quick way
%     covmatT = covmatT + cov(EEG.data(:,:,triali)');
end

% divide by N to complete the averaging
covmatT = covmatT / triali;

%% one covariance from concatenated trials

tmpdat  = reshape(EEG.data,EEG.nbchan,[]);
covmatC = cov( tmpdat' );

% same as above
% tmpdat = bsxfun(@minus,tmpdat,mean(tmpdat,2));
% covmatC = tmpdat*tmpdat'/length(tmpdat);

%% plotting 

% define color limit
clim = [-1 1]*1e3;

% plot the covariances...
figure(1), clf
subplot(121)
imagesc(covmatT)
set(gca,'clim',clim)
axis square
xlabel('Channel'), ylabel('Channel')
title('Covariance from each trial')


subplot(122)
imagesc(covmatC)
set(gca,'clim',clim)
axis square
xlabel('Channel'), ylabel('Channel')
title('Covariance from concatenated trial')


% how do the two approaches relate to each other?
figure(2), clf
plot(covmatT(:),covmatC(:),'s','markerfacecolor','k')

%% covariance based on time window

% state time windows (in ms!)
timewin1 = [ -800 0  ];
timewin2 = [  0 1500 ];

% convert to indices
tidx1 = dsearchn(EEG.times',timewin1');
tidx2 = dsearchn(EEG.times',timewin2');

% initialize matrices
[covPre,covPst] = deal( zeros(EEG.nbchan) );

% loop over trials
for triali=1:EEG.trials
    
    %%% first time window
    tmp = EEG.data(:,tidx1(1):tidx1(2),triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    covPre = covPre + tmp*tmp'/diff(tidx1);

    %%% second time window
    tmp = EEG.data(:,tidx2(1):tidx2(2),triali);
    tmp = bsxfun(@minus,tmp,mean(tmp,2));
    covPst = covPst + tmp*tmp'/diff(tidx2);
end


% and plot
figure(3), clf
subplot(121)
imagesc(covPre), axis square
title([ 'Covariance: ' num2str(timewin1(1)) ' to ' num2str(timewin1(2)) ])
% set(gca,'clim',climT)

subplot(122)
imagesc(covPst), axis square
title([ 'Covariance: ' num2str(timewin2(1)) ' to ' num2str(timewin2(2)) ])
% set(gca,'clim',climT)

%% covariance based on narrowband frequency

% filter parameters
centf = 9; % Hz
fwhm  = 5; % Hz

% narrowband filter the data
EEG.fdata = filterFGx(EEG.data,EEG.srate,centf,fwhm);

% initialize (F=filter, B=broadband)
[covF,covB] = deal( zeros(EEG.nbchan) );

% loop over trials and compute covariance matrices
for triali=1:EEG.trials
    
    % filtered data
    covF = covF + cov( EEG.fdata(:,tidx2(1):tidx2(2),triali)' );
    
    % broadband data
    covB = covB + cov( EEG.data(:,tidx2(1):tidx2(2),triali)' );
end


% color limits (note the order-of-magnitude difference!)
climB = [-1 1]*1e5;
climF = [-1 1]*1e4;

% and plot
figure(4), clf
subplot(121)
imagesc(covB), axis square
title('Broadband covariance')
set(gca,'clim',climB)

subplot(122)
imagesc(covF), axis square
title('Filtered covariance')
set(gca,'clim',climF)

%% done.
