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

% 16 subjects, each 56*260*340 (channels*points*trails)
root='data\ERN\';
listing=dir([root '*.mat']);
fnum=length(listing);
addpath('lib')

% Load data and perform CA
Xc=zeros(56,260,340*fnum);
Xr=zeros(56,260,340*fnum);
Y=nan(340*fnum,1);
ref={'riemann','logeuclid','euclid'};
for f=1:fnum
    load([root listing(f).name])
    idf=(f-1)*340+1:f*340;
    Y(idf)=y;
    Xr(:,:,idf)=x;
    [~,Xc(:,:,idf)]=centroid_align(x,ref{3});
end

N=1; bca_dte=[];
for t=1:N
    BCA=zeros(fnum,1);
    for n=1:fnum
        disp(n)
        % Single target data   
        idt=(n-1)*340+1:n*340;
        ids=1:340*fnum; ids(idt)=[];          

        % Multi source data
        Xsc=Xc(:,:,ids); Xtc=Xc(:,:,idt);
        Ys=Y(ids); Yt=Y(idt);
        idsP=Yt==1; idsN=Yt==0;
        w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);
        
        % xDAWN filtering
        [xTrain,xTest] = xDAWN(3,Xsc,Ys,Xtc);
        E=mean(xTrain(:,:,Ys==1),3); % Compute SCM by the raw source data
        Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
        Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);
        
        % Centroid Alignment
        Cs=centroid_align(Xsn,ref{1});
        Ct=centroid_align(Xtn,ref{1});
        
        % Logarithmic mapping on aligned covariance matrices
        Xs=logmap(Cs,'MI'); % dimension: 78*5100 (features*samples)
        Xt=logmap(Ct,'MI'); 

        %% MEKT
        options.d = 10;             % subspace bases 
        options.T = 5;              % iterations, default=5
        options.alpha= 0.01;        % the parameter for source discriminability
        options.beta = 0.1;         % the parameter for target locality, default=0.1
        options.rho = 20;           % the parameter for projection constraints
        options.clf = 'svm';        % the string for base classifier, 'slda' or 'svm'
        Cls=[];
        [Zs, Zt] = MEKT(Xs, Xt, Ys, Cls, options);
        model = libsvmtrain(w,Ys,Zs','-h 0 -t 0 -c 0.125');
        Ypre = libpredict(Yt,Zt',model);
        BCA(n)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
    end
    disp(mean(BCA)*100)
    bca_dte=[bca_dte,mean(BCA)*100];
end

rmpath('lib')
