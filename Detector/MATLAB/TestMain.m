clc,clear, close all
warning off;
addpath ../../Trainer/GML_AdaBoost_Matlab_Toolbox_0.3/

addpath ../../Trainer/HOG

maxObjectSize = [512,512];
minObjectSize = [128,128];
scaleFactor = 1.1;

rootpath='./';
ClassifierPath=strcat(rootpath,'dt/*.mat');
dirs=dir(ClassifierPath);
dircell=struct2cell(dirs);
classifier_names=dircell(1,:);

classifiers = cell(length(classifier_names),1);
for numStage = 1:length(classifier_names)
    % ����ÿһ��ķ�����?    
    classifiers{numStage,1} = load(['dt/Stage',num2str(numStage)]);    
end

FeatureSelected = cell(length(classifiers),1);
for k = 1:length(classifiers)
    StrongClassifier = classifiers{k,1}.StrongClassifier;
    Nodes = StrongClassifier.GLearners;
    FeatureSelected{k,1} = cell(length(Nodes),1);
    for q = 1:length(Nodes)
        CurrentNode = Nodes{1,q};
        DIM = [];
        DIM(1) = get_dim(CurrentNode);
        d = 2;
        while 1
            if ~isempty(CurrentNode.parent)
               ParentNode = CurrentNode.parent;
               CurrentNode = ParentNode;
               DIM(d) = get_dim(CurrentNode);
               d = d+1;
            else
               break;
            end
        end
        SingleNoteDim = zeros(length(DIM),5);
        for shoot = 1:length(DIM)
            Dim = DIM(shoot);
            BinInd = mod(Dim,9);
            if BinInd == 0
               BinInd =9;
            end
            if BinInd == 9
               BlockInd = floor(Dim/9);
            else
               BlockInd = floor(Dim/9)+1;
            end

            xInd = mod(BlockInd,15);
            if xInd == 0
               xInd = 15;
            end
            if xInd == 15
               yInd = BlockInd/15;
            else
               yInd = floor(BlockInd/15)+1;
            end 
               
            SingleNoteDim(shoot,:) = [BinInd BlockInd xInd yInd 1];       
        end
        FeatureSelected{k,1}{q,1} = unique(SingleNoteDim,'rows');
    end
end

ImagePath = strcat(rootpath,'TestImages/*.jpg');
dirs1=dir(ImagePath);           
dircell1=struct2cell(dirs1);      
image_names1=dircell1(1,:);

for NumImage = 1:length(image_names1)
disp(['processing image', num2str(NumImage)]);
im0 = imread([rootpath,'TestImages/',image_names1{1,NumImage}]);
im = rgb2gray(im0);
imageSize = size(im);
factor = 1;
originalWindowSize = minObjectSize;
nwin_y = 15;
nwin_x = 15;

ObjBound = zeros(imageSize);

im2 = double(im);

boxes_final = zeros(0,5);
ind = 1;

silly = 1;
while  true
    tic
    windowSize = [fix(originalWindowSize(1)*factor), fix(originalWindowSize(2)*factor)]; %windowSize = ��ⴰ��ԭͼ���еĶ�Ӧ���?    
    scaledImageSize = [fix(imageSize(1)/factor), fix(imageSize(2)/factor)];
    if scaledImageSize(1)<originalWindowSize(1) || scaledImageSize(2)<originalWindowSize(2)
        break;
    end
    processingRectSize = [scaledImageSize(1)-originalWindowSize(1)+1, scaledImageSize(2)-originalWindowSize(2)+1];
    
    if( processingRectSize(1)<= 0 || processingRectSize(2)<= 0 )
        break;
    end
    if( windowSize(1)>maxObjectSize(1) || windowSize(2)>maxObjectSize(2) )
        break;
    end
    if( windowSize(1)<minObjectSize(1) || windowSize(2)<minObjectSize(2) )
        continue;
    end
    
    scaledImage = imresize(im,scaledImageSize);
    
    xStep = 8;  %�������ڵĻ�������,x����
    yStep = 8;  %�������ڵĻ�������,y����
    
    hx = [-1,0,1];
    hy = -hx';
    
    grad_xr = imfilter(double(scaledImage),hx,'corr','replicate','same');
    grad_yu = imfilter(double(scaledImage),hy,'corr','replicate','same');
    %angle=atan2(grad_yu,grad_xr);
    angle=atan(grad_yu./(grad_xr+eps));
    grad=((grad_yu.^2)+(grad_xr.^2)).^.5;
    
    grad_integral = integral_image(grad);

    qangle_temp = uint8(floor( (angle-(-pi/2))/(pi/9)+1 )); % [-pi/2, -pi/2+pi/9), [-pi/2+pi/9, -pi/2+2*pi/9)
    mask = (qangle_temp == 10);
    qangle = qangle_temp-uint8(mask);
    
    hist = cell(9,1);
    for binIdx = 1:9
        hist{binIdx,1} = zeros(scaledImageSize);
    end
    
    for binIdx = 1:9
        mask = qangle == binIdx;
        hist{binIdx,1} = hist{binIdx,1}+mask.*grad;
    end
    
    hist_integral = cell(9,1);
    for binIdx = 1:9
        hist_integral{binIdx,1} = integral_image(hist{binIdx,1});
    end
    
    %display 'I am a silly monkey';
    for y = 1:yStep:processingRectSize(1)
        for x = 1:xStep:processingRectSize(2)
            silly = silly+1;
           
%%%%%%%%%%%%%%%%%%%%%%%%%% HOG Feature %%%%%%%%%%%%%%%%%%%%%%%%%%%                
            H1 = zeros(9*nwin_y*nwin_x,1);

            for k = 1:length(classifiers)
                for q = 1:length(FeatureSelected{k,1})                       
                    for shoot = 1:size(FeatureSelected{k,1}{q,1},1)
                                    
                        BinInd = FeatureSelected{k,1}{q,1}(shoot,1);
                        BlockInd = FeatureSelected{k,1}{q,1}(shoot,2);
                        xInd =  FeatureSelected{k,1}{q,1}(shoot,3);
                        yInd =  FeatureSelected{k,1}{q,1}(shoot,4);
                                   
                        H_norm = grad_integral(y+(yInd-1)*8,x+(xInd-1)*8)-grad_integral(y+(yInd-1)*8,x+(xInd-1)*8+16-1)-grad_integral(y+(yInd-1)*8+16-1,x+(xInd-1)*8)+grad_integral(y+(yInd-1)*8+16-1,x+(xInd-1)*8+16-1);
                        H_Temp = hist_integral{BinInd}(y+(yInd-1)*8,x+(xInd-1)*8)-hist_integral{BinInd}(y+(yInd-1)*8,x+(xInd-1)*8+16-1)-hist_integral{BinInd}(y+(yInd-1)*8+16-1,x+(xInd-1)*8)+hist_integral{BinInd}(y+(yInd-1)*8+16-1,x+(xInd-1)*8+16-1);
                        
                         H1((BlockInd-1)*9+BinInd,1) = H_Temp/(H_norm+eps);               
                     end
                 end
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                H = [H1];
                PredictResult = true;
                for k = 1:length(classifiers)
                   StrongClassifier = classifiers{k,1}.StrongClassifier;
                   Confidence = Classify(StrongClassifier.GLearners, StrongClassifier.GWeights, H);
                   GControl = sign(Confidence-classifiers{k,1}.StrongClassifier.threshold);
                   if GControl<0
                      PredictResult = false;
                      break;
                   end
                end
                 
                if  PredictResult == true
                    boxes_final(ind,:) = [x*factor,y*factor,x*factor+windowSize(2)-1,y*factor+windowSize(1)-1, Confidence];
                    ind = ind+1;
                end
        end
    end        
    factor = factor*scaleFactor;
    toc
end

pick = nms(boxes_final,0.3, 'overlap');
boxes_final = boxes_final(pick,:);
figure('visible','off')
set(gca,'position',[0 0 1 1]);
imshow(im0); axis normal;
for i = 1:size(boxes_final,1)
    a1 = max([boxes_final(i,2),1]);
    a2 = max([boxes_final(i,1),1]);
    a3 = min([boxes_final(i,4),imageSize(1)]);
    a4 = min([boxes_final(i,3),imageSize(2)]);
    
    if a3-a1<=200 && a4-a2<=200
        rectangle('Position',[boxes_final(i,1), boxes_final(i,2),   boxes_final(i,3)- boxes_final(i,1), boxes_final(i,4)- boxes_final(i,2) ], 'EdgeColor','r',  'LineWidth', 1);
%         ResultFolder = ['Result2/',image_names1{1,NumImage}(1:length(image_names1{1,NumImage})-4)];
%         if exist(ResultFolder,'dir')==0
%            mkdir(ResultFolder);
%         end
%         imwrite(im(a1:a3,a2:a4),[ResultFolder,'/',image_names1{1,NumImage}(1:length(image_names1{1,NumImage})-4),'_',num2str(i),'.jpg']);
    end

end
saveas(gcf,['Result/',image_names1{1,NumImage}(1:length(image_names1{1,NumImage})-4),'_processed.jpg'])
end