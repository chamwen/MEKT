function [Zs, Zt] = MEKT(Xs, Xt, Ys, Yt0, options)
    % =====================
    % Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces (MEKT)
    % =====================
    % Author: Wen Zhang and Dongrui Wu
    % Date: Oct. 9, 2019
    % E-mail: wenz@hust.edu.cn

    % Input: Xs and Xt: source and target features, in d*N format,
    %        Ys and Yt0: source labels and target pseudo-labels (for evaluation)
    %        Parameters: 
    %        d: subspace bases, [5,20],
    %        T: iterations, default=5,
    %        beta: the parameter for L, default=0.1,
    %        alpha: the parameter for P, [2^(-15),2^(-5)],
    %        rho: the parameter for Q, [5,40],
    %        clf: the string for base classifier, 'slda' or 'svm'.
    % Output: Embeddings Zs, Zt.

    % Set options
    d = options.d; T = options.T;
    alpha = options.alpha; beta = options.beta;
    rho = options.rho; clf = options.clf;

    % Get variable sizes
    [ms,ns] = size(Xs); [mt,nt] = size(Xt);
    class = unique(Ys); C = length(class);

    % Initialize P: source domain discriminability
    meanTotal = mean(Xs,2);
    Sw = zeros(ms);
    Sb = zeros(ms);
    for i=1:C
        Xi = Xs(:,Ys==class(i));
        meanClass = mean(Xi,2);
        Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
        Sw = Sw + Xi*Hi*Xi';
        Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
    end
    P = zeros(2*ms,2*ms); P(1:ms,1:ms) = Sw;
    P0 = zeros(2*ms,2*ms); P0(1:ms,1:ms) = Sb;

    % Initialize L: target data locality
    manifold.k = 10; % default set to 10
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'HeatKernel';
    W = lapgraph(Xt',manifold);
    D = full(diag(sum(W,2)));
    L = D-W;
    L = [zeros(ms),zeros(mt); zeros(ms),Xt*L*Xt'];

    % Initialize Q: parameter transfer and regularization |B-A|_F+|B|_F
    Q = [eye(ms),-eye(mt);-eye(ms),2*eye(mt)];

    % Initialize S: target components perservation
    Ht = eye(nt)-1/(nt)*ones(nt,nt);
    S = [zeros(ms),zeros(mt); zeros(ms),Xt*Ht*Xt'];

    for t = 1:T
        % Calculate R: joint probability distribution shift
        Ns=1/ns*onehot(Ys,unique(Ys)); Nt=zeros(nt,C);
        if ~isempty(Yt0); Nt=1/nt*onehot(Yt0,unique(Ys)); end
        M=[Ns*Ns',-Ns*Nt';-Nt*Ns',Nt*Nt'];  
        X = [Xs,zeros(size(Xt));zeros(size(Xs)),Xt];
        R = X*M*X';

        % Generalized eigendecompostion
        Emin = alpha*P +  beta*L + rho*Q + R; % alpha*P + beta*L + rho*Q + R;
        Emax = S + alpha*P0;
        [W,~] = eigs(Emin+10^(-3)*eye(ms+mt), Emax, d, 'SM'); % SM: smallestabs

        % Smallest magnitudes
        A = W(1:ms, :);
        B = W(ms+1:end, :);

        % Embeddings
        Zs = A'*Xs;
        Zt = B'*Xt;

        if t>1
            if strcmp(clf,'slda')
                % for MI balanced data
                Yt0 = slda(Zt,Zs,Ys);
            elseif strcmp(clf,'svm')
                % for ERP unbalanced data
                w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);
                model = libsvmtrain(w,Ys,Zs','-h 0 -t 0 -c 0.125');
                Yt0 = libpredict(ones(size(Zt,2),1),Zt',model);
           end
        end
    end
end

function y_onehot=onehot(y,class)
    % Encode label to onehot form
    % Input:
    % y: label vector, N*1
    % Output:
    % y_onehot: onehot label matrix, N*C

    nc=length(class);
    num=length(y);
    y_onehot=zeros(num, nc);
    for i=1:num
        idx=nc-find(class==y(i))+1;
        y_onehot(i, idx)=1;
    end
end
