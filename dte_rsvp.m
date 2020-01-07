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
for t=1:fnum
    idx=sum(nTrials(1:t-1));
    idf=idx+1:idx+nTrials(t);
    xr=xAll(:,:,idf); yr=yAll(idf);
    [~,Xc(:,:,idf)]=centroid_align(xr,ref{3});
end

bca_dte=[];
N=20; time=zeros(1,N);
for t=1:N
    BCA=zeros(fnum,1);
    tic;
    for n=1:fnum
        % Single target data
        idx=sum(nTrials(1:n-1));
        idt=idx+1:idx+nTrials(n);
        ids=1:length(yAll);
        ids(idt)=[];

        % Multi source data
        Xsc=Xc(:,:,ids); Xtc=Xc(:,:,idt);
        Ys=yAll(ids); Yt=yAll(idt);
        idsP=Yt==1; idsN=Yt==0;

        % xDAWN filtering
        [xTrain,xTest]=xDAWN(3,Xsc,Ys,Xtc);
        E=mean(xTrain(:,:,Ys==1),3);  % Compute SCM by the raw source data
        Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
        Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);
        
        % Centroid Alignment
        Cs=centroid_align(Xsn,ref{1});
        Ct=centroid_align(Xtn,ref{1});

        % Logarithmic mapping on aligned covariance matrices
        Xs=logmap(Cs,'ERP'); % dimension: 36*4385 (features*samples)
        Xt=logmap(Ct,'ERP');

        % I: random choose half
%         idSelect = randperm(10, 5);

        % sample size for each source domain
        idn=nTrials; idn(n)=[];

        % II: ROD
%         rk=nan(10,1);
%         for te=1:10
%             idx=sum(idn(1:te-1));
%             ids=idx+1:idx+idn(te);
%             rk(te)=RODKL(Xs(:,ids)',Xt',20);
%         end
%         idx = [(1:10)',rk];
%         idx = flip(sortrows(idx,2),1);
%         idSelect = sort(idx(1:5,1));

        % III: Domain transferability estimation
        rk=nan(2,fnum-1);
        for te=1:fnum-1
            idx=sum(idn(1:te-1));
            ids=idx+1:idx+idn(te);
            rk(:,te)=DTE(Xs(:,ids)',Xt',Ys(ids));
        end
        rk(1,:)=mapminmax(rk(1,:),1,0);
        rk(2,:)=mapminmax(rk(2,:),0,1);
        a=rk(1,:).*rk(2,:);
        [~,index] = sort(a,'descend');
        idSelect = index(1:5);
        
        ids=[];
        for i=1:length(idSelect)
            idx=sum(idn(1:idSelect(i)-1));
            ids=[ids,idx+1:idx+idn(idSelect(i))];
        end
        Xs=Xs(:,ids); Ys=Ys(ids);

		w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);

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
        BCA(n)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
    end
    time(t)=toc/11;
    disp(mean(BCA)*100)
    bca_dte=[bca_dte,mean(BCA)*100];
end

rmpath('lib')
