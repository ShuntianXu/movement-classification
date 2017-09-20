function [X_norm, mu, sigma] = normalizeData(X)
% normalize to mean 0 and standard deviation 1 for each feature in X

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

% mu = mean(X);
% sigma = std(X);
% 
% X_norm = (X-mu)./sigma;

end