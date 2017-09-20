function [Xtrain, ytrain, Xval, yval] = splitTrainVal(X,y,num_labels)
% split the data into training set (70%) and validation set (30%) for
% movement classification

if nargin < 3
    num_labels = range(y) + 1;
end

Xtrain = []; ytrain = [];

% Number of examples of each label for validation set
seg = floor(0.3*length(y)/num_labels);

Xval = zeros(seg*num_labels,size(X,2));
yval = zeros(seg*num_labels,1); 

for i = 1:num_labels
    X_i = X(y==i,:);
    Xval((i-1)*seg+1:i*seg,:) = X_i(1:seg,:);
    yval((i-1)*seg+1:i*seg,:) = i*ones(seg,1);
    Xtrain = [Xtrain; X_i(seg+1:end,:)];
    ytrain = [ytrain; i*ones(size(X_i,1)-seg,1)];
end

for i = 1:10
    permtr = randperm(length(ytrain));
    Xtrain = Xtrain(permtr,:); ytrain = ytrain(permtr);
    
    permval = randperm(length(yval));
    Xval = Xval(permval,:); yval = yval(permval);
end

end