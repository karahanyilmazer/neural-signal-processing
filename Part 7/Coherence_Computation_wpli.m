function Coherence_Computation_wpli(recompute)
% folderAddress='P:\4220000.03\Mem.Loc-Project\Training\Final Analysis\';
folderAddress='P:\4220000.03\Mem.Loc-Project\Training\4OpenField2\'
list=dir('P:\4220000.03\Mem.Loc-Project\Training\4OpenField2\*.mat')

nFile=length(list);

for iFile=1:nFile
    
    filename=[folderAddress list(iFile).name '_wpli' ];
    load(filename)
    
    if recompute==1
        
        disp(iFile);
        EEG= computeCF(EEG);
        save(filename,'EEG');
        
    elseif recompute==0
        
        dpacLowG{iSub}=EEG.CFresult.dpacLowG;
        centime{iSub}=EEG.CFresult.centime;
        
    end
    
end



end
function EEG=computeCF(EEG)
%%
% find the bad channels
goodchans = cellfun(@str2double,{EEG.chanlocs.labels});
chanorder = false(32,1);
chanorder(goodchans) = 1;

% replace the data
EEGfulldata = nan(32,EEG.pnts);
EEGfulldata(chanorder,:) = EEG.data;
EEG.data = EEGfulldata;
EEG.nbchan = 32;

% update the channel labels
for i=1:32
    if chanorder(i)
        EEG.chanlocs(i).labels = num2str(i);
    else
        EEG.chanlocs(i).labels = 'nan';
    end
end
%%
%%%%%%%% Computação posterior
EEG.data = double(EEG.data);

% adjust time vector
EEG.times = (0:EEG.pnts-1)/EEG.srate;
centime = dsearchn(EEG.times',(6:EEG.times(end)-6)');%center time in a slide window of 6 seconds
% EEG.times=EEG.times'

PFCchans = 1:16;
PARchans = 17:24;
HIPchans = 25:32;



% local referencing in cortex
% EEG.data(PFCchans,:) = bsxfun(@minus,EEG.data(PFCchans,:),mean(EEG.data(PFCchans,:),1));
% EEG.data(PARchans,:) = bsxfun(@minus,EEG.data(PARchans,:),mean(EEG.data(PARchans,:),1));

% specify frequency range
min_freq =  2;
max_freq = 80;
num_frex = 100;

% define frequency and time ranges
frex = linspace(min_freq,max_freq,num_frex);

% parameters for complex Morlet wavelets
wavtime  = -2:1/EEG.srate:2;
half_wav = (length(wavtime)-1)/2;
fwhms    = linspace(.5,.5,num_frex);

% FFT parameters
nWave = length(wavtime);
nData = EEG.pnts*EEG.trials;
nConv = nWave+nData-1;

% and create wavelets
cmwX = zeros(num_frex,nConv);
for fi=1:num_frex
    cmw      = exp(1i*2*pi*frex(fi).*wavtime) .* exp( (-4*log(2)*wavtime.^2) ./ (fwhms(fi)^2) );
    tempX     = fft(cmw,nConv);
    cmwX(fi,:) = tempX ./ max(tempX);
end


%% run convolution to extract tf power

% %%%% spectra of data
% dataX = fft(EEG.data,nConv,2);
%
% % initialize output time-frequency data
% tf = zeros(EEG.nbchan,num_frex,length(times2save));
% lineffect = zeros(EEG.nbchan,num_frex);
%
% for fi=1:num_frex
%
%     % run convolution
%     as = ifft(bsxfun(@times,dataX,cmwX(fi,:)),[],2);
%     as = as(:,half_wav+1:end-half_wav);
%
%     for ti=1:length(times2save)
%         tf(:,fi,ti) = mean(abs(as(:,times2saveidx(ti)-2999:times2saveidx(ti)+2999)).^2,2);
%     end
%
%     % linear effect
%     for ci=1:EEG.nbchan
%         lineffect(ci,fi) = corr(squeeze(tf(ci,fi,:)),times2save');
%     end
%
% end
%
% EEG.tf=tf;

%% component-based convolution


% get R matrices (unfiltered)
bdat = bsxfun(@minus,EEG.data(PFCchans,:),mean(EEG.data(PFCchans,:),2));
Rpfc = bdat*bdat';

bdat = bsxfun(@minus,EEG.data(PARchans,:),mean(EEG.data(PARchans,:),2));
Rpar = bdat*bdat';

bdat = bsxfun(@minus,EEG.data(HIPchans,:),mean(EEG.data(HIPchans,:),2));
Rhip = bdat*bdat';

%% initialize variables

[phasesynch,tf,wpli] = deal( zeros(3,length(centime),num_frex) );


for fi=1:num_frex
    
    %% find best component for PFC
    
    % filter data and create S matrix
    fdat = filterFGx(EEG.data(PFCchans,:),EEG.srate,frex(fi),3);
    fdat = bsxfun(@minus,fdat,mean(fdat,2));
    Spfc = fdat*fdat';
    
    % eigendecomposition
    [V,D] = eig(Spfc,Rpfc);
    [D,i] = sort(diag(D),'descend');
    V = V(:,i);
    pfccompts = hilbert( V(:,1)'*fdat );
    
    
    %% find best component for PAR
    
    % filter data and create S matrix
    fdat = filterFGx(EEG.data(PARchans,:),EEG.srate,frex(fi),3);
    fdat = bsxfun(@minus,fdat,mean(fdat,2));
    Spar = fdat*fdat';
    
    % eigendecomposition
    [V,D] = eig(Spar,Rpar);
    [D,i] = sort(diag(D),'descend');
    V = V(:,i);
    parcompts = hilbert( V(:,1)'*fdat );
    
    
    %% find best component for PAR
    
    % filter data and create S matrix
    fdat = filterFGx(EEG.data(HIPchans,:),EEG.srate,frex(fi),3);
    fdat = bsxfun(@minus,fdat,mean(fdat,2));
    Ship = fdat*fdat';
    
    % eigendecomposition
    [V,D] = eig(Ship,Rhip);
    [D,i] = sort(diag(D),'descend');
    V = V(:,i);
    hipcompts = hilbert( V(:,1)'*fdat );
    
    %% extract power in each region for TF maps
    
    tf(1,:,fi) = abs(pfccompts(centime)).^2;
    tf(2,:,fi) = abs(parcompts(centime)).^2;
    tf(3,:,fi) = abs(hipcompts(centime)).^2;
    
    %% compute synchronization
    
    %%% Loop over center time points
    for timei=1:length(centime)
        
        
        %%% standard phase synchronization
        pfc_phase = angle(pfccompts(centime(timei)-1500:centime(timei)+1500));
        par_phase = angle(parcompts(centime(timei)-1500:centime(timei)+1500));
        hip_phase = angle(hipcompts(centime(timei)-1500:centime(timei)+1500));
        
        phasesynch(1,timei,fi) = abs(mean(exp(1i*(pfc_phase-par_phase))));
        phasesynch(2,timei,fi) = abs(mean(exp(1i*(pfc_phase-hip_phase))));
        phasesynch(3,timei,fi) = abs(mean(exp(1i*(hip_phase-par_phase))));
        
        
        %%% wpli
        cdd = bsxfun(@times,pfccompts(centime(timei)-1500:centime(timei)+1500),conj(parcompts(centime(timei)-1500:centime(timei)+1500)));
        cdi = imag(cdd);
        wpli(1,timei,fi) = abs( mean( abs(cdi).*sign(cdi) ) )./mean(abs(cdi));
        
        cdd = bsxfun(@times,pfccompts(centime(timei)-1500:centime(timei)+1500),conj(hipcompts(centime(timei)-1500:centime(timei)+1500)));
        cdi = imag(cdd);
        wpli(2,timei,fi) = abs( mean( abs(cdi).*sign(cdi) ) )./mean(abs(cdi));
        
        cdd = bsxfun(@times,hipcompts(centime(timei)-1500:centime(timei)+1500),conj(parcompts(centime(timei)-1500:centime(timei)+1500)));
        cdi = imag(cdd);
        wpli(3,timei,fi) = abs( mean( abs(cdi).*sign(cdi) ) )./mean(abs(cdi));
        
    end
    
end
EEG.phasesynch=phasesynch;



end

