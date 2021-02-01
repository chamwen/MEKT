function [idx, Fs]=Fvalue(Fea, labels, Num)
% Dimensionality reduction by one-way ANOVA based on the sorted F-values
% Input:
%   Fea: d*N
%   labels: N*1
%   Num: the choosed feature numbers 
% Output:
%   idx: the indices of the choosed features 
%   FS: the choosed feature

n = size(Fea,2);
f_values = zeros(n,1);

for i=1:n
    [~,table] = anova1(Fea(:,i),labels,'off');
    f_values(i,1) = table{2,5};
end

idx = [(1:n)',f_values];
idx = flip(sortrows(idx,2),1);
idx = sort(idx(1:Num,1));
Fs = Fea(:,idx);
