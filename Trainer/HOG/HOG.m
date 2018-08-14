%Image descriptor based on Histogram of Orientated Gradients for gray-level images. This code 
%was developed for the work: O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
%Classifier-Fusion Schemes: An Application To Pedestrian Detection,' In: 12th International IEEE 
%Conference On Intelligent Transportation Systems, 2009, St. Louis, 2009. V. 1. P. 432-437. In 
%case of publication with this code, please cite the paper above.

function H=HOG(Im,nwin_x,nwin_y)
% nwin_x=15; %set the number of HOG windows per bound box
% nwin_y=15;
B=9; %set here the number of histogram bins
[L,C]=size(Im); % L num of lines ; C num of columns
H=zeros(nwin_x*nwin_y*B,1); % column vector with zeros
m=sqrt(L/2);
% if C==1 % if num of columns==1
%     Im=im_recover(Im,m,2*m);%verify the size of image, e.g. 25x50
%     L=2*m;
%     C=m;
% end
Im=double(Im);
step_x=floor(C/(nwin_x+1));
step_y=floor(L/(nwin_y+1));

hx = [-1,0,1];
hy = -hx';
grad_xr = imfilter(double(Im),hx,'corr','replicate','same');
grad_yu = imfilter(double(Im),hy,'corr','replicate','same');
    
%angles=atan2(grad_yu,grad_xr);
angles=atan(grad_yu./(grad_xr+eps));
magnit=((grad_yu.^2)+(grad_xr.^2)).^.5;
grad_integral = integral_image(magnit);

qangle_temp = uint8(floor( (angles-(-pi/2))/(pi/9)+1 )); % [-pi/2, -pi/2+pi/9), [-pi/2+pi/9, -pi/2+2*pi/9)
mask = (qangle_temp == 10);
qangle = qangle_temp-uint8(mask);

hist = cell(9,1);
for binIdx = 1:9
    hist{binIdx,1} = zeros([L,C]);
end

for binIdx = 1:9
    mask = qangle == binIdx;
    hist{binIdx,1} = hist{binIdx,1}+mask.*magnit;
end

hist_integral = cell(9,1);
for binIdx = 1:9
    hist_integral{binIdx,1} = integral_image(hist{binIdx,1});
end  

count=1;
H = zeros(9*nwin_y*nwin_x,1);
for n=0:nwin_y-1
    for m=0:nwin_x-1
        rect = [m*8+1,n*8+1,16,16];
        H_norm = grad_integral(rect(2),rect(1))-grad_integral(rect(2),rect(1)+rect(3)-1)-grad_integral(rect(2)+rect(4)-1,rect(1))+grad_integral(rect(2)+rect(4)-1,rect(1)+rect(3)-1);
        for bin = 1:9
            H_Temp = hist_integral{bin}(rect(2),rect(1))-hist_integral{bin}(rect(2),rect(1)+rect(3)-1)-hist_integral{bin}(rect(2)+rect(4)-1,rect(1))+hist_integral{bin}(rect(2)+rect(4)-1,rect(1)+rect(3)-1);
            H((count-1)*9+bin,1) = H_Temp/(H_norm+eps);
        end
        count=count+1;
    end
end

end
