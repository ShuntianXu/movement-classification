function [CM, prec, rec, F1, cr] = evaluateMetrics(yval, pval, num_labels)
% evaluation metrics, confusion matrix, precision, recall, F1, 
% and classification rate

if nargin<3
    num_labels = range(yval)+1;
end

CM = zeros(num_labels);

for i = 1:num_labels
    for j = 1:num_labels
        CM(i,j) = sum((pval==i)&(yval==j));
    end
end

prec = 100*diag(CM)./sum(CM,2);
rec = 100*diag(CM)./sum(CM)';
F1 = 2*prec.*rec./(prec+rec);
cr = 100*mean(double(pval == yval));

end