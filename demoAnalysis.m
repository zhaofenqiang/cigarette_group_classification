clc;
clear;

sub_info = importdata('D:\cigarette\sub_info2.csv');
id = sub_info.textdata(2:end,1);
label = sub_info.data(:,1);
sum_length = length(find(label == 1)) + length(find(label == 2));

X = sub_info.data(1:sum_length, 2:end);
Y = label(1:sum_length);
for i = 1: size(X, 2)
    X(:,i) = (X(:,i) - min(X(:,i)))/(max(X(:,i)) - min(X(:,i)));
end   %按列归一化
%% plot
group1_index = [];
group2_index = [];
for i = 1:length(Y)
    if label(i) == 1
        group1_index = [group1_index, i];
    else
        group2_index = [group2_index, i];
    end        
end

figure(1)
subplot(3,1,1);
for i = 1:length(group1_index)
    plot(X(group1_index(i),:),'r');hold on;
end
title('info1');
subplot(3,1,2);
for i = 1:length(group2_index)
    plot(X(group2_index(i),:),'g');hold on;
end
title('info2');
subplot(3,1,3);
for i = 1:size(X,1)
    if ismember(i,group1_index)
        plot(X(i,:),'r');hold on;
    else
        plot(X(i,:),'g');hold on;
    end
end
title('sub_info');

%%
test_index = [];
train_index = [];
for i = 1:length(Y)
    if mod(i,3) == 0
        test_index = [test_index,i];
    else
        train_index = [train_index, i];
    end
end
trainX = X(train_index, :);
trainY = Y(train_index);
testX = X(test_index, :);
testY = Y(test_index);

% trainX = [1,2;1,1;0,0;0,1;0,2;1,-1;2,0;2,-1;3,0;3,1];
% trainY = [1,1,1,1,1,2,2,2,2,2]'; 
% testX = [0.5,0.5;2,2;1.5,0.2;2,0.5];
% testY = [1,1,2,2]'; 
model = svmtrain(trainY, trainX, '-t 2');
[predicted_label,accuracy,decision_values] = svmpredict(testY,testX,model);
