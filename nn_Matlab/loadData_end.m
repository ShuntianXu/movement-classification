function [X, y] = loadData_end
% load raw data from the text files and map features for end detector
% X is the selected features of each example, y is the labels, num is the
% number of samples of each example

% load training data
n = 9 + 44 + 24;

% save the filenames
File = cell(1,n);
for i = 1:9
    File{i} = ['lunge' num2str(i) '.txt'];
end
for i = i+1:i+44
    File{i} = ['pressup' num2str(i-9) '.txt'];
end
for i = i+1:i+24
    File{i} = ['situp' num2str(i-53) '.txt'];
end

% retrieve raw data from each file
X = []; y = []; num = []; count = 1;
cd ../data/raw;
for i = 1:n % randperm(n)
    [X_temp, y_temp] = retrieveData(File{i});
    for j=1:length(X_temp)
        if size(X_temp{j},1) > 9
            Xx = X_temp{j}(end-9:end,:)';
            X = [X; Xx(:)'];
            y = [y; y_temp(j)];
        end
    end
end
cd ../../nn_Matlab;

% make data randomly ordered
l = length(y);
for i = 1:10
    perm = randperm(l);
    X = X(perm,:); y = y(perm);
end

end