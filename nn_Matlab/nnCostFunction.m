function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% nnCostFunction computes the cost and the gradient of the two layer neural 
% network for classification 

% Reshape the weights for each corresponding layer
Weight1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Weight2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of training examples
m = size(X, 1);

%% Vectorized implementation
% Convert the label of each example into vector representation
Y = zeros(num_labels,m);
for i = 1:m
    Y(y(i),i) = 1;
end

% Compute outputs and activation for each nodes
a_1 = [ones(m,1) X]';
z_2 = Weight1*a_1;
a_2 = [ones(1,m); sigmoid(z_2)];
z_3 = Weight2*a_2;
a_3 = sigmoid(z_3);

% Compute cost with regularization
J = -1/m * sum(sum(Y.*log(a_3) + (1-Y).*log(1-a_3))) + ...
    lambda/2/m * (sum(sum(Weight1(:,2:end).^2)) + sum(sum(Weight2(:,2:end).^2)));

% Backpropagation
delta_3 = a_3 - Y;
delta_2 = Weight2(:,2:end)'*delta_3.*sigmoidGradient(z_2);
Weight2_grad = delta_3*a_2';
Weight1_grad = delta_2*a_1';

Weight1_grad = Weight1_grad / m + [zeros(size(Weight1,1),1) lambda/m*Weight1(:,2:end)];
Weight2_grad = Weight2_grad / m + [zeros(size(Weight2,1),1) lambda/m*Weight2(:,2:end)];

% Unroll gradients
grad = [Weight1_grad(:); Weight2_grad(:)];


%% For-loop implementation

% % Convert the label of each example into vector representation
% Y = zeros(m,num_labels);
% for i = 1:m
%     Y(i,y(i)) = 1;
% end
% 
% % Compute outputs
% Outputs = sigmoid([ones(m,1) sigmoid([ones(m,1) X]*Weight1')]*Weight2');
% 
% % Compute cost with regularization
% J = -1/m * sum(sum(Y.*log(Outputs) + (1-Y).*log(1-Outputs))) + ...
%     lambda/2/m * (sum(sum(Weight1(:,2:end).^2)) + sum(sum(Weight2(:,2:end).^2)));
% 
% % Backpropagation
% for i = 1:m
%     a_1 = [1 X(i,: )]';
%     z_2 = Weight1*a_1;
%     a_2 = [1; sigmoid(z_2)];
%     z_3 = Weight2*a_2;
%     a_3 = sigmoid(z_3);
%     
%     delta_3 = a_3 - Y(i,:)';
%     delta_2 = Weight2(:,2:end)'*delta_3.*sigmoidGradient(z_2);
%     Weight2_grad = Weight2_grad + delta_3*a_2';
%     Weight1_grad = Weight1_grad + delta_2*a_1';
% end
% 
% Weight1_grad = Weight1_grad / m + [zeros(size(Weight1,1),1) lambda/m*Weight1(:,2:end)];
% Weight2_grad = Weight2_grad / m + [zeros(size(Weight2,1),1) lambda/m*Weight2(:,2:end)];
% 
% % Unroll gradients
% grad = [Weight1_grad(:) ; Weight2_grad(:)];

end
