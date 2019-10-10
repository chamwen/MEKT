function [fTrain,fTest] = xDAWN(nfilter,xTrain,yTrain,xTest)
    % xDAWN filtering used to reduce channel numbers
    % Reference:
    %   xDAWN algorithm to enhance evoked potentials: application to 
    %   brain-computer interface
    %   B. Rivet, A. Souloumiac, V. Attina, and G. Gibert
    %   IEEE Trans. on Biomedical Engineering, Aug. 2009.

    % Calculates the average evoqued potential for each class
    P0 = mean(xTrain(:,:,yTrain==0),3);
    P1 = mean(xTrain(:,:,yTrain==1),3);
     
    % Estimates the covariances matrices
    C0 = cov(P0'); % of the class 0
    C1 = cov(P1'); % of the class 1
    C = cov(xTrain(:,:)'); % of the signal

    % Calcultates the spatial filters
    [V0,~] = eig(C\C0);
    [V1,~] = eig(C\C1);
    V = [V0(:,1:nfilter), V1(:,1:nfilter)];
	
	% For each trial, apply the spatial filter
	fTrain = zeros(size(V,2),size(xTrain,2),size(xTrain,3));
    for k=1:size(xTrain,3)
        fTrain(:,:,k) = V'*xTrain(:,:,k);
    end
	
	fTest = zeros(size(V,2),size(xTest,2),size(xTest,3));
    for k=1:size(xTest,3)
        fTest(:,:,k) = V'*xTest(:,:,k);
    end
end
   