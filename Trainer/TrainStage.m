function [flag,StrongClassifier] = TrainStage(NumStage,PosFeature,NegFeature,MaxIter,minHitRate,maxFalseAlarm)
PosData = zeros(length(PosFeature{1,1}),length(PosFeature));
NegData = zeros(length(NegFeature{1,1}),length(NegFeature));
for k = 1:length(PosFeature)
    temp = PosFeature{k,1};
    PosData(:,k) = temp;
end
for k = 1:length(NegFeature)
    temp = NegFeature{k,1};
    NegData(:,k) = temp;
end
TrainData = zeros(length(PosFeature{1,1}),length(PosFeature)+length(NegFeature));
TrainLabels = zeros(1,length(PosFeature)+length(NegFeature));
l1 = 1;
l2 = 1;
for k = 1:length(PosFeature)+length(NegFeature)
    if (mod(k,2) == 1 && k<length(PosFeature)*2)
        TrainData(:,k) = PosData(:,l1);
        TrainLabels(1,k) = 1;
        l1 = l1+1;
    elseif (mod(k,2) == 0 && k<length(NegFeature)*2) || k>length(PosFeature)*2
        TrainData(:,k) = NegData(:,l2);
        TrainLabels(1,k) = -1;
        l2 = l2+1;
    end
end


falseAlarm = zeros(1, MaxIter);
hitRate = zeros(1, MaxIter);
weak_learner = tree_node_w(3); % pass the number of tree splits to the constructor
GLearners = [];
GWeights = [];
threshold = 0;
lrn_num = 1;
while lrn_num<=MaxIter
    if lrn_num == 1
        %training gentle adaboost
        [GLearners, GWeights] = GentleAdaBoost(weak_learner, TrainData, TrainLabels, 1, GWeights, GLearners);
        
        %evaluating control error
        Confidence = Classify(GLearners, GWeights, TrainData);
        thresholdIdx = 1;
        %thresholdIdx = ceil((1.0 - minHitRate) * length(PosFeature));
        ConfidencePos = zeros(1,length(PosFeature));
        l = 1;
        for k = 1:length(Confidence)
            if TrainLabels(1,k) == 1
                ConfidencePos(1,l) = Confidence(1,k);
                l = l+1;
            end
        end
        ConfidencePos2 = sort(ConfidencePos);
        threshold = ConfidencePos2(thresholdIdx);
        GControl = sign(Confidence-threshold);
                
        NumFA = 0;
        NumTP = 0;
        for k = 1:length(PosFeature)+length(NegFeature)
            if GControl(k) ~= -1 && TrainLabels(k) == -1
                NumFA = NumFA+1;
            end
        end
        for k = 1:length(PosFeature)+length(NegFeature)
            if (GControl(k) == 1 || GControl(k) == 0)&& TrainLabels(k) == 1
                NumTP = NumTP+1;
            end
        end
        falseAlarm(lrn_num) = double(falseAlarm(lrn_num) + NumFA / length(NegFeature));
        hitRate(lrn_num) = double(hitRate(lrn_num) + NumTP / length(PosFeature));
        lrn_num = lrn_num+1;
    elseif falseAlarm(lrn_num-1)>=maxFalseAlarm
       fprintf('Iter %d, The hitRate of stage %d is currently %f \n',lrn_num-1,NumStage,  hitRate(lrn_num-1));
       fprintf('Iter %d, The falseAlarmRate of stage %d is currently %f \n',lrn_num-1,NumStage,  falseAlarm(lrn_num-1));
       %training gentle adaboost
       [GLearners, GWeights] = GentleAdaBoost(weak_learner, TrainData, TrainLabels, 1, GWeights, GLearners);
       
       %evaluating control error
       Confidence = Classify(GLearners, GWeights, TrainData);
       thresholdIdx = 1;
       %thresholdIdx = ceil((1.0 - minHitRate) * length(PosFeature));
       ConfidencePos = zeros(1,length(PosFeature));
       l = 1;
       for k = 1:length(Confidence)
           if TrainLabels(1,k) == 1
               ConfidencePos(1,l) = Confidence(1,k);
               l = l+1;
           end
       end
       ConfidencePos2 = sort(ConfidencePos);
       threshold = ConfidencePos2(thresholdIdx);
       GControl = sign(Confidence-threshold);
       
       NumFA = 0;
       NumTP = 0;
       for k = 1:length(PosFeature)+length(NegFeature)
           if GControl(k)~= -1 && TrainLabels(k) == -1
               NumFA = NumFA+1;
           end
       end
       for k = 1:length(PosFeature)+length(NegFeature)
           if (GControl(k) == 1 || GControl(k) == 0)&& TrainLabels(k) == 1
               NumTP = NumTP+1;
           end
       end
       falseAlarm(lrn_num) = falseAlarm(lrn_num) + NumFA / length(NegFeature);
       hitRate(lrn_num) = hitRate(lrn_num) + NumTP / length(PosFeature);
       lrn_num = lrn_num+1;      
    elseif falseAlarm(lrn_num-1)<maxFalseAlarm
       fprintf('Iter %d,The hitRate of stage %d is currently %f \n',lrn_num-1,NumStage, hitRate(lrn_num-1));
       fprintf('Iter %d,The falseAlarmRate of stage %d is currently %f \n',lrn_num-1, NumStage, falseAlarm(lrn_num-1));
       break
    end
end
if lrn_num>MaxIter
   flag = true; % all the stages can be correctly classified by this stage
else
   flag = false;
end

StrongClassifier.NumStage = NumStage;
StrongClassifier.GLearners = GLearners;
StrongClassifier.GWeights = GWeights;
StrongClassifier.threshold = threshold;

end