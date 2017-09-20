function [X, y, num] = loadData
% load raw data from the text files and map features for movement
% classifier
% X is the selected features of each example, y is the labels, num is the
% number of samples of each example

% total number of raw data text files, 9 lunge, 44 press up, 24 sit up
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
for i = 1:n
    [X_temp, y_temp] = retrieveData(File{i});
    X = [X; mapFeature(X_temp)];
    y = [y; y_temp'];
    
    % get the number of samples of acceleration data in each example
    for j = 1:length(X_temp)
        num(count) = size(X_temp{j},1);
        count = count + 1;
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