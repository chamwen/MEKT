function rk = RODKL(Xs,Xt,dim)
% Rank of Domain
% Ref: 
%    Geodesic Flow Kernel for Unsupervised Domain Adaptation. 
%    B. Gong, Y. Shi, F. Sha, and K. Grauman.  
%    Proceedings of the CVPR, Providence, RI, June 2012.
% Contact:
%    Boqing Gong (boqinggo@usc.edu)

% Xs: N_s\times D
Ps = pca(Xs);
Pt = pca(Xt);

Q = [Ps,null(Ps')];
Pt = Pt(:,1:dim);
QPt = Q' * Pt;
[~,~,~,Gam] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
theta = real(acos(diag(Gam)));

KL=nan(dim,1);
for i=1:dim
    sigmas = (Xs*Ps(:,i))'*Xs*Ps(:,i)/size(Xs,1);
    sigmat = (Xt*Pt(:,i))'*Xt*Pt(:,i)/size(Xt,1);     
    kldis = (sigmas^4+sigmat^4)/(2*sigmas^2*sigmat^2)-1;
    KL(i) = theta(i)*kldis;
end
rk = sum(KL)/dim;
end
