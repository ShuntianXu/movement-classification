function [CEtrain, CEval, Jtrain, Jval] = learningCurve(input_layer_size,...
                                          hidden_layer_size, num_labels,...
                                          Xt, yt, Xval, yval, lambda, max_iter)

% plot learning curve
% Xt and yt are the training set, Xval and yval are the validation set
% lambda is the regularization parameter, max_iter is the maximum iteration
% of the training
                                      
x_axis = [25:25:1375, 1383]; % number of examples for training
CEtrain = zeros(1,length(x_axis));
CEval = CEtrain;
Jtrain = CEtrain;
Jval = CEval;
count = 1;

for i = x_axis
    
    % subset of training set
    Xtrain = Xt(1:i,:); ytrain = yt(1:i,:);
    
    % Train NN
    [Weight1, Weight2] = trainNN(input_layer_size, hidden_layer_size, ...
                         num_labels, Xtrain, ytrain, lambda, max_iter);

    % cost of training and validation
    nn_params = [Weight1(:); Weight2(:)];
    Jtrain(count) = nnCostFunction(nn_params, input_layer_size, ...
                    hidden_layer_size, num_labels, Xtrain, ytrain, 0);
    Jval(count) = nnCostFunction(nn_params, input_layer_size, ...
                  hidden_layer_size, num_labels, Xval, yval, 0);
                             
    % classification error of training and validation
    ptrain = predict(Weight1, Weight2, Xtrain);
    pval = predict(Weight1, Weight2, Xval);
    
    CEtrain(count) = 1-mean(double(ptrain == ytrain));
    CEval(count) = 1-mean(double(pval == yval));
    
    count = count + 1;
end


plot(x_axis, Jtrain, 'b', x_axis, Jval, 'r', 'lineWidth', 1.5);
xlabel('Number of training examples', 'FontSize', 18);
ylabel('Error', 'FontSize', 18);
lgd = legend('Train', 'Validation');
lgd.FontSize = 18;   

figure;
plot(x_axis, CEtrain, 'b', x_axis, CEval, 'r', 'lineWidth', 1.5);
xlabel('Number of training examples', 'FontSize', 18);
ylabel('Classification Error %', 'FontSize', 18);
lgd = legend('Train', 'Validation');
lgd.FontSize = 18;   

end