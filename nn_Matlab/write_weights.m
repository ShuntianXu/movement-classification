function write_weights(filename, X)
% save the trained weights into file

fid = fopen(filename, 'w+t');

if fid < 0
   fprintf('error opening file\n');
   return;
end

for i=1:size(X,1)
    for j=1:size(X,2)
        if j==1
            fprintf(fid, '{');
        end
        if j~=size(X,2)
            if X(i,j) >= 0
                fprintf(fid,' %8.6f, ', X(i,j));  % %8.6f
            else
                fprintf(fid,'%8.6f, ', X(i,j));   % %5.0d %6.0d
            end
        else
            if X(i,j) >= 0
                fprintf(fid,' %8.6f', X(i,j));  % %8.6f
            else
                fprintf(fid,'%8.6f', X(i,j));   % %5.0d %6.0d
            end
        end
        
        if j==size(X,2) && i~=size(X,1)
            fprintf(fid, '},');
        end
        if j==size(X,2) && i==size(X,1)
            fprintf(fid, '}');
        end
    end
    fprintf(fid, '\n');
end

fclose(fid);