function [Ytest,d,C] = mdm(COVtest,COVtrain,Ytrain)
    % Minimum distance to mean (MDM) classifier
    % Reference:
    %   Covariance Toolbox by Alexandre Barachant
    %   https://github.com/alexandrebarachant/covariancetoolbox
    
    labels = unique(Ytrain);
    Nclass = length(labels);
    C = cell(Nclass,1);
    
    % estimation of center
    for i=1:Nclass
        C{i} = riemann_mean(COVtrain(:,:,Ytrain==labels(i)));
    end

    % classification
    NTesttrial = size(COVtest,3);
    
    d = zeros(NTesttrial,Nclass);
    for j=1:NTesttrial
        for i=1:Nclass
            d(j,i) = distance_riemann(COVtest(:,:,j),C{i});
        end
    end
    
    [~,ix] = min(d,[],2);
    Ytest = labels(ix);
    
end

function a = distance_riemann(A,B)
    a = sqrt(sum(log(eig(A,B)).^2));
end