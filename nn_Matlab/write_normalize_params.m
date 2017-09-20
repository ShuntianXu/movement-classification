function write_normalize_params(filename, mu, sigma)
% save normalized parameters, i.e. mean and standard deviation

fid = fopen(filename, 'w+t');

if fid < 0
   fprintf('error opening file\n');
   return;
end

fprintf(fid, 'Mean\n');
for i = 1:length(mu)
    if mu(i) >= 0
        fprintf(fid,' %6.1f, ', mu(i));
    else
        fprintf(fid,' %6.1f, ', mu(i));
    end
end
fprintf(fid, '\n\nStandard Deviation\n');
for i = 1:length(sigma)
    if sigma(i) >= 0
        fprintf(fid,' %6.1f, ', sigma(i));
    else
        fprintf(fid,' %6.1f, ', sigma(i));
    end
end


fclose(fid);