%% Start Detector using ANN

clear; close all; clc;

%% Data clean and process for training
[X, y] = loadData_start;
[Xtrain, ytrain, Xval, yval] = splitTrainVal_binary(X,y);

%% Data Normalization
[Xtrain, mu, sigma] = normalizeData(Xtrain);
Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);

%% NN Architecture initialization
input_layer_size  = 36;
hidden_layer_size = 18;
num_labels = 2;
lambda = 0.1;
max_iter = 600;

%% Learning Curve
% [CEtr, CEval, Jtr, Jval] = learningCurve(input_layer_size, hidden_layer_size, num_labels, ...
%                                Xtrain, ytrain, Xval, yval, lambda, max_iter);

%% Train NN
[Weight1, Weight2] = trainNN(input_layer_size, hidden_layer_size, num_labels, ...
                             Xtrain, ytrain, lambda, max_iter);

%% Validation
ptrain = predict(Weight1, Weight2, Xtrain);
pval = predict(Weight1, Weight2, Xval);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(ptrain == ytrain)) * 100);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(pval == yval)) * 100);

[CM, pre, rec, F1, cr] = evaluateMetrics(yval, pval, num_labels);

