clc,clear,close all

addpath ../Trainer/GML_AdaBoost_Matlab_Toolbox_0.3
addpath ../Trainer/HOG

rootpath='./';

fprintf('Generating Sample Lists\n'); 

PosSamplePath=strcat(rootpath,'pos_norm/*.jpg');
dirs1=dir(PosSamplePath);           
dircell1=struct2cell(dirs1);      
image_names1=dircell1(1,:);
fprintf('Positive Sample List Generated\n'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NegSampleSourcePath=strcat(rootpath,'NegSampleSource/*.jpg');
dirs2=dir(NegSampleSourcePath);           
dircell2=struct2cell(dirs2);
image_names2=dircell2(1,:);
fprintf('Negative Sample Source List Generated\n'); 

ForbiddenAreas = cell(0,2);
n = 0;
fid0 = fopen('ForbiddenAreaList.txt','r');
while ~feof(fid0)                
    tline=fgetl(fid0);           
    if strcmp( tline(length(tline)-3:length(tline)),'.jpg')
       n = n+1;
       ForbiddenAreas{n,1} = tline;
       n2 = 1;
    else
       n2 = n2+1;
       for i = 1:4
           ForbiddenAreas{n,n2} = str2num(tline);
       end
    end       
end
fclose(fid0);

Map = -1*ones(length(image_names2),1);
for l = 1:length(image_names2)  
    for k = 1:size(ForbiddenAreas,1)
        if strcmp(ForbiddenAreas(k,1),image_names2{1,l})
           Map(l) = k;
        end
    end
end


MaxNumStage = 15;
NumPos = 315;
NumNeg = 1000;
StageMaxIter = 10000;

ClassifierPath=strcat(rootpath,'dt/*.mat');
dirs=dir(ClassifierPath);
dircell=struct2cell(dirs);
classifier_names=dircell(1,:);

CascadeClassifer = cell(MaxNumStage,1);
if ~isempty(classifier_names)   
    for numStage = 1:length(classifier_names)
        CascadeClassifer{numStage,1} = load(['dt/Stage',num2str(numStage)]);    
    end
end

for NumStage = 1:MaxNumStage
    ResultFolder = ['neg_norm_batches/Batch',num2str(NumStage)];
    if exist(ResultFolder,'dir')==0
       mkdir(ResultFolder);
    end
    
    H_Pos = cell(NumPos,1);
    NumGeneratedPos = 1;
    for l =  1:length(image_names1)
        fprintf('%d Images are Consumed for Generating Positive Samples\n', l);
        im = imread([rootpath,'pos_norm/',image_names1{1,l}]);
        im = imresize(im,[128,128]);
        H_Pos{NumGeneratedPos,1} = HOG(im,15,15);
        NumGeneratedPos = NumGeneratedPos+1;
        if NumGeneratedPos>NumPos
            break;
        end
    end
    
    H_Neg = cell(NumNeg,1);
    skip = false(length(image_names2),1);
    
    if NumStage == 1
       NumGeneratedNeg = 1;
       l = 1;
       NumOfConsumedImage = 0;
       while NumGeneratedNeg<=NumNeg
           if l == length(image_names2)
               l = 1;
           end
           im = imread([rootpath,'NegSampleSource/',image_names2{1,l}]);
           im = rgb2gray(im);
           if skip(l) == true
              l = l+1;
              continue;
           end
           la = 0;
           while 1
                la = la+1;
                if la>100000
                    skip(l) = true;
                    if ~any(~skip)
                        error('Not Enough Samples for Generating Negative Samples\n');
                    end
                    break;
                end
                p1 = 128+(size(im,1)-128)*rand(1,1);  % The size of the sample is set to [128, 128]
                p2 = 128+(size(im,2)-128)*rand(1,1);
                q1 = 128+(size(im,1)-128)*rand(1,1); % window_location(ï¿½ï¿½ï¿½Ï½Ç¶ï¿½ï¿½ï¿½ï¿½Î»ï¿½ï¿?yï¿½ï¿½ï¿?
                q2 = 128+(size(im,2)-128)*rand(1,1); % window_location(ï¿½ï¿½ï¿½Ï½Ç¶ï¿½ï¿½ï¿½ï¿½Î»ï¿½ï¿?xï¿½ï¿½ï¿?
                if q1+p1>size(im,1) || q2+p2>size(im,2)
                   continue
                end
                NumOfConsumedImage = NumOfConsumedImage+1;
                flag = false;
                if Map(l)~=-1
                    for m = 1:size(ForbiddenAreas,2)
                        if ~isempty(ForbiddenAreas{Map(l),m})
                           overlapY = p1 + ForbiddenAreas{Map(l),m}(3) - ...
                           (  max([q1+p1, ForbiddenAreas{Map(l),m}(1)+ForbiddenAreas{Map(l),m}(3)]) ...
                               - min([q1, ForbiddenAreas{Map(l),m}(1)]) );
                           overlapX = p2 + ForbiddenAreas{Map(l),m}(4) - ...
                           (  max([q2+p2, ForbiddenAreas{Map(l),m}(2)+ForbiddenAreas{Map(l),m}(4)]) ...
                               - min([q2, ForbiddenAreas{Map(l),m}(2)]) );
                           if (overlapX>0) && (overlapY>0) && ...
                                   ( (p1<=3*ForbiddenAreas{Map(l),m}(3)) || (p2<=3*ForbiddenAreas{Map(l),m}(4)) )
                               flag = true;
                           end
                        end
                    end
                end
                if flag == true
                    continue
                end
                im_crop = im( max([q1,1]):min([q1+p1,size(im,1)]), max([q2,1]):min([q2+p2,size(im,2)]) );
                im_crop = imresize(im_crop,[128,128]);      
                H_Neg{NumGeneratedNeg,1} = HOG(im_crop,15,15);
                imwrite(im_crop,[ResultFolder,'/',num2str(NumGeneratedNeg),'.jpg']);
                NumGeneratedNeg = NumGeneratedNeg+1;
                fprintf('%d  negative samples have been generated\n', NumGeneratedNeg-1);
                break;                
           end
           l = l+1;      
       end
    else
       NumGeneratedNeg = 1;
       l = 1;
       NumOfConsumedImage = 0;
       while NumGeneratedNeg<=NumNeg
           if l == length(image_names2)
               l = 1;
           end
           im = imread([rootpath,'NegSampleSource/',image_names2{1,l}]);
           im = rgb2gray(im);
           if skip(l) == true
              l = l+1;
              continue;
           end
           la = 0;
           while 1
                la = la+1;
                if la>100000
                    skip(l) = true;
                    if ~any(~skip)
                        error('Not Enough Samples for Generating Negative Samples\n');
                    end
                    break;
                end
                p1 = 128+(size(im,1)-128)*rand(1,1);  % The size of the sample is set to [128, 128]
                p2 = 128+(size(im,2)-128)*rand(1,1);
                q1 = 128+(size(im,1)-128)*rand(1,1); % window_location(ï¿½ï¿½ï¿½Ï½Ç¶ï¿½ï¿½ï¿½ï¿½Î»ï¿½ï¿?yï¿½ï¿½ï¿?
                q2 = 128+(size(im,2)-128)*rand(1,1); % window_location(ï¿½ï¿½ï¿½Ï½Ç¶ï¿½ï¿½ï¿½ï¿½Î»ï¿½ï¿?xï¿½ï¿½ï¿?
                if q1+p1>size(im,1) || q2+p2>size(im,2)
                   continue
                end
                NumOfConsumedImage = NumOfConsumedImage+1;
                if mod(NumOfConsumedImage,100) == 0
                   disp([num2str(NumOfConsumedImage) ,'  sub-images are consumed for Generating Negative Samples\n']);
                end
                flag = false;
                if Map(l)~=-1
                    for m = 1:size(ForbiddenAreas,2)
                        if ~isempty(ForbiddenAreas{Map(l),m})
                           overlapY = p1 + ForbiddenAreas{Map(l),m}(3) - ...
                           (  max([q1+p1, ForbiddenAreas{Map(l),m}(1)+ForbiddenAreas{Map(l),m}(3)]) ...
                               - min([q1, ForbiddenAreas{Map(l),m}(1)]) );
                           overlapX = p2 + ForbiddenAreas{Map(l),m}(4) - ...
                           (  max([q2+p2, ForbiddenAreas{Map(l),m}(2)+ForbiddenAreas{Map(l),m}(4)]) ...
                               - min([q2, ForbiddenAreas{Map(l),m}(2)]) );
                           if (overlapX>0) && (overlapY>0) && ...
                                   ( (p1<=3*ForbiddenAreas{Map(l),m}(3)) || (p2<=3*ForbiddenAreas{Map(l),m}(4)) )
                               flag = true;
                           end
                        end
                    end
                end
                if flag == true
                    continue
                end
                im_crop = im( max([q1,1]):min([q1+p1,size(im,1)]), max([q2,1]):min([q2+p2,size(im,2)]) );
                im_crop = imresize(im_crop,[128,128]);
                H = HOG(im_crop,15,15);
                if predict(H,CascadeClassifer)
                   H_Neg{NumGeneratedNeg,1} = H;
                   imwrite(im_crop,[ResultFolder,'/',num2str(NumGeneratedNeg),'.jpg']);
                   NumGeneratedNeg = NumGeneratedNeg+1;
                   fprintf('%d  negative samples have been generated\n', NumGeneratedNeg-1);
                else
                   continue;
                end
                break;                
           end
           l = l+1;         
       end
    end
    

   [flag,StrongClassifier] = TrainStage(NumStage,H_Pos,H_Neg,StageMaxIter,0.995,0.5);
   CascadeClassifer{NumStage,1} = StrongClassifier;
   

    save(['dt/Stage',num2str(NumStage)], 'StrongClassifier')
end
