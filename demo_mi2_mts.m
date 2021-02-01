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
% 9 subjects, each 22*750*144 (channels*points*trails)
root='data\MI2\';
listing=dir([root '*.mat']);
addpath('lib');

% Load data and perform congruent transform
fnum=length(listing);
Ca=nan(22,22,144*fnum);
Xr=nan(22,750,144*9);
Xa=nan(22,750,144*9);
Y=nan(144*fnum,1);
ref={'riemann','logeuclid','euclid'};
for f=1:fnum
    load([root listing(f).name])
    idf=(f-1)*144+1:f*144;
    Y(idf) = y; Xr(:,:,idf) = x;
    Ca(:,:,idf) = centroid_align(x,ref{1});
    [~,Xa(:,:,idf)] = centroid_align(x,ref{3});
end
    
N=1; bca_dte=[];
for t=1:N

    BCA=zeros(fnum,1);
    for n=1:fnum
        disp(n)
        % Single target data & multi source data
        idt=(n-1)*144+1:n*144;
        ids=1:144*fnum; ids(idt)=[];             
        Yt=Y(idt); Ys=Y(ids);
        idsP=Yt==1; idsN=Yt==0;
        Ct=Ca(:,:,idt);  Cs=Ca(:,:,ids);

        % Logarithmic mapping on aligned covariance matrices
        Xs=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
        Xt=logmap(Ct,'MI');

        %% MEKT
        options.d = 10;             % subspace bases 
        options.T = 5;              % iterations, default=5
        options.alpha= 0.01;        % the parameter for source discriminability
        options.beta = 0.1;         % the parameter for target locality, default=0.1
        options.rho = 20;           % the parameter for subspace discrepancy
        options.clf = 'slda';        % the string for base classifier, 'slda' or 'svm'
        Cls = [];
        [Zs, Zt] = MEKT(Xs, Xt, Ys, Cls, options);
        Ypre = slda(Zt,Zs,Ys);
        BCA(n)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
        clear options
    end
    disp(mean(BCA)*100)
    bca_dte=[bca_dte,mean(BCA)*100];
end

rmpath('lib');
