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
% 7 subjects, each 59*300*200 (channels*points*trails)
root='data\MI1\';
listing=dir([root '*.mat']);
addpath('lib');

fnum=length(listing);
BCA=zeros(fnum,fnum-1);
ref={'riemann','logeuclid','euclid'};
for tr=1:fnum
    disp(tr)
    % Single target data
    load([root listing(tr).name])
    Xtr=x; Yt=y;
    tes=1:fnum; tes(tr)=[];

    for te=1:fnum-1
        % Single source data
        load([root listing(tes(te)).name])
        Xsr=x; Ys=y;
        idsP=Yt==1; idsN=Yt==0;
        
        % Centroid Alignment
        Cs=centroid_align(Xsr,ref{1});
        Ct=centroid_align(Xtr,ref{1});

        % Logarithmic mapping on aligned covariance matrices
        Xs=logmap(Cs,'MI'); % dimension: 1770*200 (features*samples)
        Xt=logmap(Ct,'MI');
        
        % Dimensionality reduction by one-way ANOVA based on F-values
        [idx, Fs]=Fvalue(Xs', Ys, length(Ys));
        Xs=Fs'; Xt=Xt(idx,:);

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
        BCA(tr,te)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
    end
end
disp(mean(mean(BCA,1),2)*100')

rmpath('lib');
