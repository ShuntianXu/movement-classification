function [Xtrain, ytrain, Xval, yval] = splitTrainVal_binary(X,y)
% split data into training and validation for start and end detectors
% the number of selected examples of each class are proportional 
% to the ratio of corresponding total number of examples of that class over
% the total number of examples of all classes
% 

% number of examples of each label for validation set

Xval = zeros(296, size(X,2));
yval = zeros(296, 1); 

X_1 = X(y==1,:); X_2 = X(y==2,:); X_3 = X(y==3,:); X_4 = X(y==4,:);
Xval(1:148,:) = X_4(1:148,:);
Xval(149:188,:) = X_1(1:40,:);
Xval(189:243,:) = X_2(1:55,:);
Xval(244:296,:) = X_1(1:53,:);
yval(1:148,:) = 2*ones(148,1); % invalid - 2
yval(149:296,:) = ones(148,1); % valid - 1

Xtrain = [X_4(149:end,:); X_1(41:136,:); X_2(56:182,:); X_3(54:178,:)];
ytrain = ones(size(Xtrain,1),1); % valid - 1
ytrain(1:(size(X_4,1)-148)) = 2; % invalid - 2


for i = 1:10
    permtr = randperm(length(ytrain));
    Xtrain = Xtrain(permtr,:); ytrain = ytrain(permtr);
    
    permval = randperm(length(yval));
    Xval = Xval(permval,:); yval = yval(permval);
end

end