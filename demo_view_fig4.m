% =====================
% Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces (MEKT)
% =====================
% Author: Wen Zhang and Dongrui Wu
% Date: Oct. 9, 2019
% E-mail: wenz@hust.edu.cn

clc; clear all; close all; warning off;

% Load datasets: 
% 9 subjects, each 22*750*144 (channels*trails*samples)
root = 'data\MI2\';
listing = dir([root '*.mat']);
addpath('lib');

fnum = length(listing);
ref = {'riemann','logeuclid','euclid'};

BCA = zeros(fnum,fnum-1);
for tr=1
    % Single target data
    load([root listing(tr).name])
    Xtr = x; Yt = y;
    tes = 1:fnum; tes(tr) = [];

    for te=2
        % Single source data
        load([root listing(tes(te)).name])
        Xsr = x; Ys = y;
        idsP=Yt==1; idsN=Yt==0;

        Cs = centroid_align(Xsr,ref{1});
        Ct = centroid_align(Xtr,ref{1});

        ns = length(Ys); c=unique(Ys);
        sizes0 = 2*ones([length(find(Ys==c(1))),1]);
        sizes1 = 10*ones([length(find(Ys==c(2))),1]);
        sizet0 = 2*ones([length(find(Yt==c(1))),1]);
        sizet1 = 10*ones([length(find(Yt==c(2))),1]);

        % Logarithmic mapping
        Xs = logmap(Cs,'MI'); % dimension: 253*144 (features*samples)
        Xt = logmap(Ct,'MI');

        %% MEKT
        options.d = 10;             % subspace bases 
        options.T = 5;              % iterations, default=5
        options.alpha= 0.01;        % the parameter for source discriminability
        options.beta = 0.1;         % the parameter for target locality, default=0.1
        options.rho = 20;           % the parameter for subspace discrepancy
        options.clf = 'lda';        % the string for base classifier, 'lda' or 'svm'
        Cls = [];
        [Zs, Zt] = MEKT(Xs, Xt, Ys, Cls, options);
        Ypre = slda(Zt,Zs,Ys);
        BCA(tr,te)=mean(Yt==Ypre);

        % Visualization MEKT
        ftsne = tsne([Zs';Zt']);
        figure; set(gcf,'position',[300, 200, 800, 400])
        ts=ftsne(1:ns,:); ts0=ts(Ys==c(1),:); ts1=ts(Ys==c(2),:);
        tt=ftsne(ns+1:end,:); tt0=tt(Yt==c(1),:); tt1=tt(Yt==c(2),:);
        scatter(ts0(:,1),ts0(:,2),sizes0,'b','filled'), hold on
        scatter(ts1(:,1),ts1(:,2),sizes1,'b*'), hold on
        scatter(tt0(:,1),tt0(:,2),sizet0,'r','filled'), hold on
        scatter(tt1(:,1),tt1(:,2),sizet1,'r*'), hold off
        xlabel('z1'); ylabel('z2'); title('MEKT-R')
        set(gca,'FontSize', 14, 'Fontname', 'Times New Roman');                     
        box on
        axis square
        str = cellstr(['Source class 1';'Source class 2';'Target class 1';'Target class 2']);
        legend(str,'location','EastOutside','fontsize', 14,'Fontname','Times New Roman');
    end
end
disp(['MI2, S2-->S1: ', num2str(BCA(tr,te)*100)])

rmpath('lib');
