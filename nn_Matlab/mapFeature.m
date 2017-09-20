function Input = mapFeature(X)
% map the raw data of each example to selected features, including max, 
% min, std, mean

Input = zeros(length(X), 12);
for i = 1:length(X)
    Input(i,:) = [max(X{i}), min(X{i}), mean(X{i}), std(X{i})];
end

end