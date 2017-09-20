function g = sigmoidGradient(z)
% gradient of sigmoid function

g = sigmoid(z).*(1-sigmoid(z));

end
