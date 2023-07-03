function [accuracy, recall, precision, fMeasure] = classifierPerformance(groundTruth, estimation)
% classfierPerformance Assess the performace of a binary classfier
% groundTruth and estimation: logical vector or matrix
tp = sum(groundTruth&estimation, 'all');
tn = sum((~groundTruth)&(~estimation),'all');
fp = sum((~groundTruth)&estimation,'all');
fn = sum(groundTruth&(~estimation),'all');
accuracy = (tp + tn)/(tp + tn + fp + fn);
recall = tp/(tp + fn);
precision = tp/(tp + fp);
fMeasure = 2*recall*precision/(recall + precision);
end
