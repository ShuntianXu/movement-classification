function p = predict(Weight1, Weight2, X)
% predict the label of a new example given the weights of a trained neural
% network

m = size(X, 1);
h1 = sigmoid([ones(m, 1) X] * Weight1');
h2 = sigmoid([ones(m, 1) h1] * Weight2');
[~, p] = max(h2, [], 2);

end
