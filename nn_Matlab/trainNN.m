function [W1, W2, J] = trainNN(input_layer_size, hidden_layer_size, ...
                               num_labels, Xtrain, ytrain, lambda, max_iter)
% train neural network

% randomly initialize weights
initial_Weight1 = initializeWeights(input_layer_size, hidden_layer_size);
initial_Weight2 = initializeWeights(hidden_layer_size, num_labels);

% unroll parameters as input for optimization function fmincg
initial_nn_params = [initial_Weight1(:) ; initial_Weight2(:)];

% training ANN
options = optimset('MaxIter', max_iter);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda);
                               
[nn_params, J] = fmincg(costFunction, initial_nn_params, options);

W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                       
end