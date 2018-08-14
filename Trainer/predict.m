function PredictResult = predict(H,CascadeClassifier)
PredictResult = true;
for k = 1:size(CascadeClassifier,1)
    if ~isempty(CascadeClassifier{k,1})
        StrongClassifier = CascadeClassifier{k,1};
        if isfield(StrongClassifier, 'StrongClassifier')
           Confidence = Classify(StrongClassifier.StrongClassifier.GLearners, StrongClassifier.StrongClassifier.GWeights, H);
           GControl = sign(Confidence-StrongClassifier.StrongClassifier.threshold);
        else
           Confidence = Classify(StrongClassifier.GLearners, StrongClassifier.GWeights, H);
           GControl = sign(Confidence-StrongClassifier.threshold);
        end
        if GControl<0
            PredictResult = false;
            break;            
        end
    end
end