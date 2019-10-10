% =====================
% Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces (MEKT)
% =====================
% Author: Wen Zhang and Dongrui Wu
% Date: Oct. 9, 2019
% E-mail: wenz@hust.edu.cn

clc;
clear all;
close all;
warning off;

% Load datasets: 
% 11 subjects, each 8*45*n (channels*points*trails)
load('data\RSVP.mat')
addpath('lib')

% Load data and perform congruent transform
fnum=length(nTrials);
[m,n,~]=size(xAll);
Xc=zeros(m,n,length(yAll));
ref={'riemann','logeuclid','euclid'};
for k=1:fnum
    idx=sum(nTrials(1:k-1));
    idf=idx+1:idx+nTrials(k);
    xr=xAll(:,:,idf); yr=yAll(idf);
    [~,Xc(:,:,idf)]=centroid_align(xr,ref{3});
end
    
N=1; time=zeros(1,N);
for t=1:N
    
    tic;
    BCA=zeros(fnum,fnum-1);
    for tr=1:fnum
        disp(tr)
        % Single target data
        idx=sum(nTrials(1:tr-1));
        idt=idx+1:idx+nTrials(tr);
        Xtc=Xc(:,:,idt); Yt=yAll(idt);
        tes=1:11; tes(tr)=[];

        for te=1:fnum-1
            % Single source data
            id=sum(nTrials(1:tes(te)-1));
            ids=id+1:id+nTrials(tes(te));
            Xsc=Xc(:,:,ids); Ys=yAll(ids);
            ns=length(Ys); nt=length(Yt); c=unique(Ys);
            idsP=Yt==1; idsN=Yt==0;
            w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);

            % xDAWN filtering
            [xTrain,xTest]=xDAWN(3,Xsc,Ys,Xtc);
            E=mean(xTrain(:,:,Ys==1),3);  % Compute SCM by the raw source data
            Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
            Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);
            
            % Centroid Alignment
            Cs=centroid_align(Xsn,ref{1});
            Ct=centroid_align(Xtn,ref{1});

            % Logarithmic mapping on aligned covariance matrices
            Xs=logmap(Cs,'ERP'); % dimension: 64*4385 (features*samples)
            Xt=logmap(Ct,'ERP');

            %% MEKT
            options.d = 10;             % subspace bases 
            options.T = 5;              % iterations, default=5
            options.alpha= 0.01;        % the parameter for source discriminability
            options.beta = 0.1;         % the parameter for target locality, default=0.1
            options.rho = 20;           % the parameter for subspace discrepancy
            options.clf = 'svm';        % the string for base classifier, 'slda' or 'svm'
            Cls=[];
            [Zs, Zt] = MEKT(Xs, Xt, Ys, Cls, options);
            model = libsvmtrain(w,Ys,Zs','-h 0 -t 0 -c 0.125');
            Ypre = libpredict(Yt,Zt',model);
            BCA(tr,te)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
        end
    end
    time(t)=toc/110;
    disp(mean(mean(BCA,1),2)*100')
end

rmpath('lib')
