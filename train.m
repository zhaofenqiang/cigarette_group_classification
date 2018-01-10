clc;
clear;

sub_info = importdata('D:\cigarette\sub_info2.csv');
id = sub_info.textdata(2:end,1);
label = sub_info.data(:,1);
label(find(label == 2)) = 1;

raw_lh = importdata('D:\cigarette\aparc_a2009s__stats_lh_all_area.txt');
lh = raw_lh.data;
lh_id = raw_lh.textdata(2:end, 1);

raw_rh = importdata('D:\cigarette\aparc_a2009s__stats_rh_all_area.txt');
rh = raw_rh.data;
rh_id = raw_rh.textdata(2:end, 1);

lh_rh = [lh, rh];
lh_rh(2,:) = [];
lh_rh = lh_rh/6000;

% raw_lh_volume = importdata('D:\cigarette\aparc_a2009s__stats_lh_all_volume.txt');
% lh_volume = raw_lh_volume.data;
% lh_volume(2,:) = [];
% lh_volume_id = raw_lh_volume.textdata(2:end, 1);

% norm_lh_rh = mapminmax(lh_rh, 0, 1); %按行归一化
% norm_lh_rh = lh_rh/4.219;
% norm_lh_volume = lh_volume / 21204;

%% relief降维
[ranked,weights] = relieff(norm_lh_volume,label,10, 'method', 'classification');
bar(weights(ranked))
select_feature_index = ranked(1:15);
select_feature = norm_lh_rh(:,select_feature_index);

%% PCA降维
[coeff,score,latent] = pca(lh_rh); 
select_feature = score(:,1:10);
%% 获取index
group1_index = [];
group2_index = [];
group3_index = [];
for i = 1:length(label)
    if label(i) == 1
        group1_index = [group1_index, i];
    elseif label(i) == 2
        group2_index = [group2_index, i];
    else
        group3_index = [group3_index, i];
    end        
end

test_index = [];
train_index = [];
for i = 1:length(label)
    if mod(i,3) == 0
        test_index = [test_index,i];
    else
        train_index = [train_index, i];
    end
end
trainX = select_feature(train_index, :);
trainY = label(train_index);
testX = select_feature(test_index, :);
testY = label(test_index);
%% train and predict
model = svmtrain(trainY, trainX, '-t 2');
[predicted_label,accuracy,decision_values] = svmpredict(testY,testX,model);

figure(1)
subplot(4,1,1);
for i = 1:length(group1_index)
    plot(norm_lh_volume(group1_index(i),:),'r');hold on;
end
title('lhvolume1');
subplot(4,1,2);
for i = 1:length(group2_index)
    plot(norm_lh_volume(group2_index(i),:),'g');hold on;
end
title('lhvolume2');
subplot(4,1,3);
for i = 1:length(group3_index)
    plot(norm_lh_volume(group3_index(i),:),'b');hold on;
end
title('lhvolume3');
subplot(4,1,4);
for i = 1:size(norm_lh_volume,1)
    if ismember(i,group1_index)
        plot(norm_lh_volume(i,:),'r');hold on;
    elseif ismember(i,group2_index)
        plot(norm_lh_volume(i,:),'g');hold on;
    else
        plot(norm_lh_volume(i,:),'b');hold on;
    end
end
title('lhvolumeall');