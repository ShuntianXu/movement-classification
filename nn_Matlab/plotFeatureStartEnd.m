%% Plot the features for start and end detector (first and last sample)

[Xstart, ystart] = loadData_start;
Xstart1 = Xstart(ystart~=4,:);
Xstart2 = Xstart(ystart==4,:);

[Xend, yend] = loadData_end;
Xend1 = Xend(yend~=4,:);
Xend2 = Xend(yend==4,:);

scatter3(Xstart1(:,1),Xstart1(:,2),Xstart1(:,3), 20, 'ro');
hold on;
scatter3(Xstart2(:,1),Xstart2(:,2),Xstart2(:,3), 20, 'bx');
xlabel('x-1', 'FontSize', 18);
ylabel('y-1', 'FontSize', 18);
zlabel('z-1', 'FontSize', 18);
lgd = legend('Valid', 'Invalid');
lgd.FontSize = 18;
title('\fontsize{18}Scatter plot of first sample');
hold off;

figure;
scatter3(Xend1(:,end-2),Xend1(:,end-1),Xend1(:,end), 20, 'ro');
hold on;
scatter3(Xend2(:,end-2),Xend2(:,end-1),Xend2(:,end), 20, 'bx');
xlabel('x-end', 'FontSize', 18);
ylabel('y-end', 'FontSize', 18);
zlabel('z-end', 'FontSize', 18);
lgd = legend('valid', 'Invalid');
lgd.FontSize = 18;
title('\fontsize{18}Scatter plot of last sample');
hold off;



