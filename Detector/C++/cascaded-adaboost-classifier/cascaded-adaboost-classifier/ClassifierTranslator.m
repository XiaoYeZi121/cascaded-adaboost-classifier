clc,clear;
% convert the classifier saved in .mat files to .txt files, to make it readable by C++
addpath ../../../../Trainer/GML_AdaBoost_Matlab_Toolbox_0.3

rootpath='./';
ClassifierPath=strcat(rootpath,'dt_MATLAB/*.mat');
dirs=dir(ClassifierPath);
dircell=struct2cell(dirs);
classifier_names=dircell(1,:);

classifiers = cell(length(classifier_names),1);
for numStage = 1:length(classifier_names) 
    classifier = load(['dt_MATLAB/Stage',num2str(numStage)]);
    classifiers{numStage, 1} = classifier;
    GLearners = classifier.StrongClassifier.GLearners;
    GWeights = classifier.StrongClassifier.GWeights;
    classifier_name_new = ['Stage',num2str(numStage),'.txt'];
    fid = fopen(['./dt/',classifier_name_new],'w');
    TranslateToC(GLearners, GWeights, fid);
    fclose(fid);
    threshold_file_name = ['Stage',num2str(numStage),'_Threshold.txt'];
    threshold = classifier.StrongClassifier.threshold;
    fid2 = fopen(['./dt2/',threshold_file_name],'w');
    fprintf(fid2, num2str(threshold));
    fclose(fid2);
end
