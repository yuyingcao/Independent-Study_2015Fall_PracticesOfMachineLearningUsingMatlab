%% load data and initialize

clear;
load linear_regression;
[m,n] = size(trainX);

Xset=[trainX,ones(m,1)];
Yset = trainY';

%% divide dataset
% divide all data into two sets
% make sure for each label, there are about equal amount of data in each set

Xset1=[];   
Xset2=[];
Y1=[];
Y2=[];

for i=1:10
    index = find(trainY'==i);
    index1 = index(1:round(end/2));
    index2 = index(round(end/2)+1:end);
    Xset1 = [Xset1; Xset(index1,1:end)];
    Xset2 = [Xset2; Xset(index2,1:end)];
    Y1 = [Y1; Yset(index1,1:end)];
    Y2 = [Y2; Yset(index2,1:end)];
end

%% crossvalidation to calculate the best theta

theta = logspace(-5,5,11);
len = length(theta);

w1 = zeros(n+1,len);
w2 = zeros(n+1,len);

error1 = zeros(len,1);
error2 = zeros(len,1);

for i = 1:length(theta)
    w1(:,i) = inv(Xset1'*Xset1+theta(i)*eye(n+1))*Xset1'*Y1;
    predict2 = Xset2*w1(:,i);
    error1(i) = sum((predict2-Y2).^2);
end

for i = 1:length(theta)
    w2(:,i) = inv(Xset2'*Xset2+theta(i)*eye(n+1))*Xset2'*Y2;
    predict1 = Xset1*w2(:,i);
    error2(i) = sum((predict1-Y1).^2);
end

total_error = error1+error2;
t_index = find(total_error==min(total_error));
best_theta = theta(t_index);
disp(['Best theta:',num2str(best_theta)]);


%% calculate best w for each label

ww = zeros(n+1,10);

for label = 1:10
    label_index = find(trainY'==label);
    y_label = zeros(size(Yset));
    y_label(label_index) = 1;
    ww(:,label) = inv(Xset'*Xset + best_theta*eye(n+1))*Xset'*y_label;
end

%% test accuracy on the training data set

Y = Xset * ww;
predictY = zeros(size(Yset));
correct = 0;

% set the column with the maximum value to be the label value
for j = 1:length(predictY)
    Y_row = Y(j,:);
    predictY(j) = find(Y_row==max(Y_row));
    if predictY(j)==Yset(j)
        correct = correct + 1;
    end
end

accuracy = correct/length(Yset);
disp(['Accuracy on training set:',num2str(accuracy*100),'%.']);


%% caculate the result for the test set

Xtest = [testX,ones(m,1)];
Ytest = Xtest*ww;

% this is the predict result for the test set
result = zeros(size(Xtest,1),1);
count = zeros(1,10);
for k = 1:length(result)
    Y_row = Ytest(k,:);
    result(k) = find(Y_row==max(Y_row));
    count(result(k)) = count(result(k))+1;
end

% show the rough distribution of the predicted result
figure;
bar([1:10],count);
title('distribution of predicted results on the test set');