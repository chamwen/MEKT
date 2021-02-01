function rank=DTE(Xs,Xt,Ys)
    % Domain transferability estimation (DTE)
    % Input:
    %   Xs and Xt: source and target features , in d*N format
    %   Ys: source labels
    % Output:
    %   rank: [difference of Ds and Dt, discriminability of Ds]
    
    % Author: Wen Zhang and Dongrui Wu
    % Date: Oct. 9, 2019
    % E-mail: wenz@hust.edu.cn
    
    X=[Xs;Xt];
    Y=[zeros(size(Xs,1),1); ones(size(Xt,1),1)];
    Sb = scatter_matrix(X,Y);
    rank(1) = norm(Sb,1); % Difference of Ds and Dt
    
    Sb = scatter_matrix(Xs,Ys);
    rank(2) = norm(Sb,1); % Discriminability of Ds
end

function Sb = scatter_matrix(X,Y) % Compute between class scatter matrix
    [classes, ~, Y] = unique(Y);
    nc = length(classes);

    Sb = zeros(size(X, 2));
    M=mean(X); n=zeros(nc,1); C=cell(nc,1);
    for i=1:nc
        cur_X = X(Y==i,:);
        n(i)=size(cur_X,1);
        C{i}=mean(cur_X);
        Sb=Sb+n(i)*(C{i}-M)'*(C{i}-M);
    end

    Sb(isnan(Sb)) = 0;
    Sb(isinf(Sb)) = 0;
end