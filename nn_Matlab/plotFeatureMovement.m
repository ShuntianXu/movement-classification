%% Plot the features for movement classification (max, min, mean, std)

[X, y] = loadData;

X1 = X(y==1,:);
X2 = X(y==2,:);
X3 = X(y==3,:);
X4 = X(y==4,:);

% plot the maximum of x,y,z
scatter3(X1(:,1),X1(:,2),X1(:,3), 20, 'ro');
hold on;
scatter3(X2(:,1),X2(:,2),X2(:,3), 20, 'gx');
scatter3(X3(:,1),X3(:,2),X3(:,3), 20, 'm+');
scatter3(X4(:,1),X4(:,2),X4(:,3), 20, 'bs');
xlabel('x-max', 'FontSize', 18);
ylabel('y-max', 'FontSize', 18);
zlabel('z-max', 'FontSize', 18);
lgd = legend('Press up', 'Sit up', 'Lunge', 'Invalid');
lgd.FontSize = 18;
hold off;

% plot the minimum of x,y,z
figure;
scatter3(X1(:,4),X1(:,5),X1(:,6), 20, 'ro');
hold on;
scatter3(X2(:,4),X2(:,5),X2(:,6), 20, 'gx');
scatter3(X3(:,4),X3(:,5),X3(:,6), 20, 'm+');
scatter3(X4(:,4),X4(:,5),X4(:,6), 20, 'bs');
xlabel('x-min', 'FontSize', 18);
ylabel('y-min', 'FontSize', 18);
zlabel('z-min', 'FontSize', 18);
lgd = legend('Press up', 'Sit up', 'Lunge', 'Invalid');
lgd.FontSize = 18;
hold off;

% plot the mean of x,y,z
figure;
scatter3(X1(:,7),X1(:,8),X1(:,9), 20, 'ro');
hold on;
scatter3(X2(:,7),X2(:,8),X2(:,9), 20, 'gx');
scatter3(X3(:,7),X3(:,8),X3(:,9), 20, 'm+');
scatter3(X4(:,7),X4(:,8),X4(:,9), 20, 'bs');
xlabel('x-mean', 'FontSize', 18);
ylabel('y-mean', 'FontSize', 18);
zlabel('z-mean', 'FontSize', 18);
lgd = legend('Press up', 'Sit up', 'Lunge', 'Invalid');
lgd.FontSize = 18;
hold off;

% plot the standard deviation of x,y,z
figure;
scatter3(X1(:,10),X1(:,11),X1(:,12), 20, 'ro');
hold on;
scatter3(X2(:,10),X2(:,11),X2(:,12), 20, 'gx');
scatter3(X3(:,10),X3(:,11),X3(:,12), 20, 'm+');
scatter3(X4(:,10),X4(:,11),X4(:,12), 20, 'bs');
xlabel('x-mean', 'FontSize', 18);
ylabel('y-mean', 'FontSize', 18);
zlabel('z-mean', 'FontSize', 18);
lgd = legend('Press up', 'Sit up', 'Lunge', 'Invalid');
lgd.FontSize = 18;
hold off;



