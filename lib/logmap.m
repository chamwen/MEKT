function Fea = logmap(COV,type)
% Logarithmic mapping on centralized signal covariance matrices
% Input:
%   COV: K*K*N, centralized signal covariance matrices
% Output:
%   Fea: tangent space features, d*N

% Author: Wen Zhang and Dongrui Wu
% Date: Oct. 9, 2019
% E-mail: wenz@hust.edu.cn
    
NTrial = size(COV,3);
N_elec = size(COV,1);

if strcmp(type,'ERP')

    % Select upper right elements related to temporal information
    N = N_elec/2;
    Fea = zeros(N*N,NTrial);
    for i=1:size(COV,3)
        Tn = logm(COV(:,:,i));
        Fea(:,i)=reshape(Tn(1:N,(N+1):end),[],1);
    end 
elseif strcmp(type,'MI')

    % Select upper triangular elements related to spatial information
    Fea = zeros(N_elec*(N_elec+1)/2,NTrial);
    index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==1;
    for i=1:NTrial
        Tn = logm(COV(:,:,i));
        tmp = reshape(sqrt(2)*triu(Tn,1)+diag(diag(Tn)),N_elec*N_elec,1);
        Fea(:,i) = tmp(index);
    end
end
