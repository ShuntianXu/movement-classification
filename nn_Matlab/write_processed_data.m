function write_processed_data(filename, X, y)
% save pre-processed data into file

fid = fopen(filename, 'w+t');

if fid < 0
   fprintf('error opening file\n');
   return;
end

for i=1:size(X,1)
    for j=1:size(X,2)
        if X(i,j) >= 0
            fprintf(fid,' %8.6f ', X(i,j));  % %8.6f
        else
            fprintf(fid,'%8.6f ', X(i,j));   % %5.0d %6.0d
        end
    end
    fprintf(fid, ' %d\n', y(i));
end

fclose(fid);