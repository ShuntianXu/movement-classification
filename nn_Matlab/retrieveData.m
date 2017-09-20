function [X, y] = retrieveData(filename)
% retrieve raw data from the text files
% filename is the name of the text file that stores the raw data
% X is the cell whose entries store the raw data of each example,
% y is the corresponding label of each example

fid = fopen(filename, 'rt');
if fid < 0
    error('error opening %s\n', filename);
end

count = 1;
line{count} = fgetl(fid);
while ischar(line{count})
    count = count + 1;
    line{count} = fgetl(fid);
end
fclose(fid);

n_x = 1;
n_xr = 1;
n_y = 1;
record = false;
for i = 1:length(line)
    if ~record && strcmp(line{i}, 'Repetition Start')
        record = true;
    elseif record && strcmp(line{i}, 'Repetition End')
        record = false;
        y(n_y) = sscanf(line{i+1}, '%d');
        n_y = n_y + 1;
        n_x = n_x + 1;
        n_xr = 1;
    elseif record
        X{n_x}(n_xr,1:3) = sscanf(line{i}, '%d');
        n_xr = n_xr + 1;
    end
end

% 1. press-up 2. sit-up 3. lunge 4.invalid
y(y==0) = 4;
if ~isempty(strfind(filename,'situp'))
    y(y==1) = 2;
elseif ~isempty(strfind(filename,'lunge'))
    y(y==1) = 3;
end

end