%% End Detector using ANN

clear; close all; clc;

%% Data clean and process for training
[X, y] = loadData_end;
[Xtrain, ytrain, Xval, yval] = splitTrainVal_binary(X,y);

%% Data Normalization
[Xtrain, mu, sigma] = normalizeData(Xtrain);
Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);

%% NN Architecture initialization
input_layer_size  = 30;
hidden_layer_size = 15;
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
ptrain = predict(single(Weight1), single(Weight2), single(Xtrain));
pval = predict(single(Weight1), single(Weight2), single(Xval));

fprintf('\nTraining Set Accuracy: %f\n', mean(double(ptrain == ytrain)) * 100);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(pval == yval)) * 100);

[CM, pre, rec, F1, cr] = evaluateMetrics(yval, pval, num_labels);

%% Weights with low precision
% If using 16-bit int, -32768~32767, Theta1*1000 is still ok is Theta1 < 10
m = size(Xval, 1);
h1 = sigmoid([ones(m, 1) Xval] * single(int8(10*Weight1)'*0.1));
h2 = sigmoid(([ones(m, 1) h1] * single(int8(10*Weight2))')*0.1);
[~, p] = max(h2, [], 2);
cr_p = 100*mean(p==yval);
