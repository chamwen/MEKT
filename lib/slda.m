function Ytest = slda(Xt,Xs,Ys)
    % Linear discriminative analysis using optimal shrinkage estimate.
    % Input:
    %   Xs and Xt: source and target features , in d*N format
    %   Ys: source labels 
    % Output:
    %   Ytest: the evaluated labels of test data
    % Reference:
    %   Covariance Toolbox by Alexandre Barachant
    %   https://github.com/alexandrebarachant/covariancetoolbox

    labels = unique(Ys);
    Nclass = length(labels);
    dim = size(Xs,1);
    mu = zeros(dim,Nclass);
    n = zeros(Nclass,1);
    Covclass = zeros(dim,dim,Nclass);

    % Compute St
    for i=1:Nclass
        Xi = Xs(:,Ys==labels(i));
        n(i) = size(Xi,2);
        mu(:,i) = mean(Xi,2);
        Covclass(:,:,i) = covariances(Xi); % Shrinkage Estimate
    end
    St = mean(Covclass,3)*(length(Ys)-1);

    % Compute Sb
    M = mean(Xs,2);
    Sb = zeros(dim);
    for i=1:Nclass
        Sb = Sb + (mu(:,i)-M)*(mu(:,i)-M)';
    end

    % Perform eigendecomposition
    [W,Lambda] = eig(Sb,St);
    [~, Index] = sort(diag(Lambda),'descend');

    % Choose the largest eigvector
    nt=length(Ys(Ys==labels(2)));
    W = W(:,Index(1));
    b = W'*sum(mu,2)*nt/length(Ys);
    s = sign(W'*mu(:,2)-b);
    
    % Classification
    y1 = s*(W'*Xt-b);
    Ytest = labels((y1>0) + 1);
end

function COV = covariances(X)
    [Ne , ~, Nt] = size(X);
    COV = zeros(Ne,Ne,Nt);
    for i=1:Nt; COV(:,:,i) = shcovft(X(:,:,i)'); end 
end

function [s, lam] = shcovft(x) % Shrinkage estimate of a covariance matrix
    p = size(x,2);
    if p==1, s=var(x); return; end

    v = var(x);
    dsv = diag(sqrt(v));
    [r, lam] = corshrink(x);
    s = dsv*r*dsv;
end

function [Rhat, lambda] = corshrink(x)
    p = size(x,2);
    sx = zscore(x);

    [r, vr] = varcov(sx);
    offdiagsumrij2 = sum(sum(tril(r,-1).^2));
    offdiagsumvrij = sum(sum(tril(vr,-1)));
    lambda = offdiagsumvrij/offdiagsumrij2;
    lambda = min(lambda, 1); lambda = max(lambda, 0);
    Rhat = (1-lambda)*r;
    Rhat(logical(eye(p))) = 1;
end

function [S, VS] = varcov(x)
    [n,p] = size(x);
    xc = x - ones(size(x,1), 1)*mean(x);
    S = cov(xc);

    tmp=nan(p,p,n);
    for i=1:n
        c=repmat(xc(i,:),[p 1]);
        tmp(:,:,i)=c'*c/p;
        clear c
    end
    VS = var(tmp, 0, 3) * n /((n-1)^2);
end
   