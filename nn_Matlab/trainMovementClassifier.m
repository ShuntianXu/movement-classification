%% Movement Classifier using ANN

clear; close all; clc;

%% Data clean and process for training
[X, y, no_xyz] = loadData;
[Xtrain, ytrain, Xval, yval] = splitTrainVal(X,y);

%% Number of acceleration samples in each example
no_xyz_stats = [];
for ii = min(no_xyz):max(no_xyz)
    if sum(no_xyz==ii)~=0
        no_xyz_stats = [no_xyz_stats; ii sum(no_xyz==ii)];
    end 
end

%% Data Normalization
[Xtrain, mu, sigma] = normalizeData(Xtrain);
Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);


%% NN Architecture initialization
input_layer_size  = 12;
hidden_layer_size = 12;  
num_labels = 4;
lambda = 0.1;
max_iter = 400;


%% Learning Curve
% [CEtr, CEval, Jtr, Jval] = learningCurve(input_layer_size, ...
%                            hidden_layer_size, num_labels, Xtrain, ...
%                            ytrain, Xval, yval, lambda, max_iter);
          
                         
%% Train NN
[Weight1, Weight2] = trainNN(input_layer_size, hidden_layer_size, ...
                             num_labels, Xtrain, ytrain, lambda, max_iter);

%% Validation
ptrain = predict(Weight1, Weight2, Xtrain);
pval = predict(Weight1, Weight2, Xval);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(ptrain == ytrain)) * 100);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(pval == yval)) * 100);

[CM, pre, rec, F1, cr] = evaluateMetrics(yval, pval, num_labels);

%% Weights with low precision
% test the effect of the weights with lower precision on the accuracy of
% the neural network
pval_lp = predict(single(int8(10*Weight1))*0.1, single(int8(10*Weight2))*0.1, Xval);
[CM_lp, pre_lp, rec_lp, F1_lp, cr_lp] = evaluateMetrics(yval, pval_lp, num_labels);

