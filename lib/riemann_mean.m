function A = riemann_mean(B,args)
    % Compute Riemannian mean of SPD matrices
    % Input:
    %   B: SPD matrices, K*K*N, K is the dimension, N is sample number
    % Output:
    %   A: Riemannian mean of the N SPD matrices, K*K
    % Reference:
    %   Covariance Toolbox by Alexandre Barachant
    %   https://github.com/alexandrebarachant/covariancetoolbox

    N_itermax = 100;
    if (nargin<2)||(isempty(args))
        tol = 10^-5;
        A = mean(B,3);
    else
        tol = args{1};
        A = args{2};
    end

    niter = 0;
    fc = 0;

    while (niter<N_itermax)
        niter = niter+1;
        % Tangent space mapping
        T = Tangent_space(B,A);
        % Sum of the squared distance
        fcn = sum(sum(T.^2));
        % Improvement
        conv = abs((fcn-fc)/fc);
        if conv<tol % break if the improvement is below the tolerance
           break;
        end
        % Arithmetic mean in tangent space
        TA = mean(T,2);
        % Back to the manifold
        A = UnTangent_space(TA,A);
        fc = fcn;
    end

    if niter==N_itermax
        disp('Warning : Nombre niter reachs maximum atteint');
    end
    critere = fc;
end

function [Feat,C] = Tangent_space(COV,C)
    NTrial = size(COV,3);
    N_elec = size(COV,1);
    Feat = zeros(N_elec*(N_elec+1)/2,NTrial);

    if nargin<2
        C = riemann_mean(COV);
    end

    index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==1;
    P = C^(-1/2);

    for i=1:NTrial
        Tn = logm(P*COV(:,:,i)*P);
        tmp = reshape(sqrt(2)*triu(Tn,1)+diag(diag(Tn)),N_elec*N_elec,1);
        Feat(:,i) = tmp(index);
    end
end

function COV = UnTangent_space(T,C)
    NTrial = size(T,2);
    N_elec = (sqrt(1+8*size(T,1))-1)/2;
    COV = zeros(N_elec,N_elec,NTrial);

    if nargin<2
        C = riemann_mean(COV);
    end

    index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==0;
    Out = zeros(N_elec*N_elec,NTrial);
    Out(not(index),:) = T;
    
    P = C^0.5;
    for i=1:NTrial
      tmp = reshape(Out(:,i),N_elec,N_elec,[]);
      tmp =  diag(diag(tmp))+triu(tmp,1)/sqrt(2) + triu(tmp,1)'/sqrt(2);
      tmp = P*tmp*P;
      COV(:,:,i) = RiemannExpMap(C,tmp);
    end
end

function Out = RiemannExpMap(P,X)

    [U,Delta] = eig(P);
    G = U*sqrt(Delta);
    Y = inv(G)*X*inv(G)';
    [V,Sigma] = eig(Y);
    Out = (G*V)*diag(exp(diag(Sigma)))*(G*V)';
end