SamplePath=strcat('/home/ecestudent/CodeblocksProject/MyCascadeDetector/WaitASecond/Exam/*.jpg');
dirs=dir(SamplePath);
dircell=struct2cell(dirs);
sample_names=dircell(1,:);

for l = 1:length(sample_names) 
    im = imread(['/home/ecestudent/CodeblocksProject/MyCascadeDetector/WaitASecond/Exam/', sample_names{1,l}]);
    im2 = rgb2gray(im);
    imwrite(im2, ['/home/ecestudent/CodeblocksProject/MyCascadeDetector/Exam/', sample_names{1,l}])
end