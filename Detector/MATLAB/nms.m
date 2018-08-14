%% NMS:non maximum suppression
function pick = nms(boxes,threshold,type)
% boxes: m x 5,��ʾ��m����5�зֱ���[x1 y1 x2 y2 score]
% threshold: IOU��ֵ
% type��IOU��ֵ�Ķ�������

    % ����Ϊ�գ���ֱ�ӷ���
    if isempty(boxes)
      pick = [];
      return;
    end

    % ����ȡ�����ϽǺ����½������Լ��������÷�(���Ŷ�)
    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,3);
    y2 = boxes(:,4);
    s = boxes(:,5);

    % ����ÿһ��������
    area = (x2-x1+1) .* (y2-y1+1);

    %���÷���������
    [vals, I] = sort(s);

    %��ʼ��
    pick = s*0;
    counter = 1;

    % ѭ��ֱ�����п������
    while ~isempty(I)
        last = length(I); %��ǰʣ��������
        i = I(last);%ѡ�����һ�������÷���ߵĿ�
        pick(counter) = i;
        counter = counter + 1;  

        %�����ཻ���
        xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = min(y2(i), y2(I(1:last-1)));  
        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1); 
        inter = w.*h;

        %��ͬ�����µ�IOU
        if strcmp(type,'Min')
            %�ص��������С������ı�ֵ
            o = inter ./ min(area(i),area(I(1:last-1)));
        else
            %����/����
            o = inter ./ (area(i) + area(I(1:last-1)) - inter);
        end

        %���������ص����С����ֵ�Ŀ������´δ���
        I = I(find(o<=threshold));
    end
    pick = pick(1:(counter-1));
end