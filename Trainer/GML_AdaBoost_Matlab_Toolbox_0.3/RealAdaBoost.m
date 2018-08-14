%   The algorithms implemented by Alexander Vezhnevets aka Vezhnick
%   <a>href="mailto:vezhnick@gmail.com">vezhnick@gmail.com</a>
%
%   Copyright (C) 2005, Vezhnevets Alexander
%   vezhnick@gmail.com
%   
%   This file is part of GML Matlab Toolbox
%   For conditions of distribution and use, see the accompanying License.txt file.
%
%   RealAdaBoost Implements boosting process based on "Real AdaBoost"
%   algorithm
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
%    [Learners, Weights, final_hyp] = RealAdaBoost(WeakLrn, Data, Labels,
%    Max_Iter, OldW, OldLrn, final_hyp)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           WeakLrn   - weak learner
%           Data      - training data. Should be DxN matrix, where D is the
%                       dimensionality of data, and N is the number of
%                       training samples.
%           Labels    - training labels. Should be 1xN matrix, where N is
%                       the number of training samples.
%           Max_Iter  - number of iterations
%           OldW      - weights of already built commitee (used for training 
%                       of already built commitee)
%           OldLrn    - learnenrs of already built commitee (used for training 
%                       of already built commitee)
%           final_hyp - output for training data of already built commitee 
%                       (used to speed up training of already built commitee)
%    Return:
%           Learners  - cell array of constructed learners 
%           Weights   - weights of learners
%           final_hyp - output for training data

function [Learners, Weights, final_hyp] = RealAdaBoost(WeakLrn, Data, Labels, Max_Iter, OldW, OldLrn, final_hyp)

if( nargin == 4)
  Learners = {};
  Weights = [];
  distr = ones(1, length(Data)) / length(Data);  %distr为样本权重
  %length(Data)返回的是矩阵Data的行数或列数（返回的是值较大的那一个），根据adaboost原理，此处必须返回样本数，所以只有当样本数大于样本维数时，程序才能正常运行
  final_hyp = zeros(1, length(Data));
elseif( nargin > 5)
  Learners = OldLrn;
  Weights = OldW;
  if(nargin < 7)
    final_hyp = Classify(Learners, Weights, Data);
  end
  distr = exp(- (Labels .* final_hyp));  
  distr = distr / sum(distr);  
else
  error('Function takes eather 4 or 6 arguments');
end


for It = 1 : Max_Iter
  
  %chose best learner

  nodes = train(WeakLrn, Data, Labels, distr);

  for i = 1:length(nodes)
    curr_tr = nodes{i};
    
    step_out = calc_output(curr_tr, Data);  
    % calc_output Implements classification of input by a classification tree node,
    % step_out(i) returns +1 if Data(:,i) belongs to the tree node, returns -1 otherwise
      
    s1 = sum( (Labels ==  1) .* (step_out) .* distr);
    s2 = sum( (Labels == -1) .* (step_out) .* distr);
    % 令 A = [1,2,3,3,8,3]; 表达式“A == 3”返回 ans = [0,0,1,1,0,1]

    if(s1 == 0 && s2 == 0)
        continue;
    end
    
    Alpha = 0.5*log((s1 + eps) / (s2+eps)); % eps为MATLAB中的无穷小量，用于防止分母为0， 注意Real AdaBoost的权重调整方式与Descrete AdaBoost（也就是最基本的adaboost）不同

    Weights(end+1) = Alpha;
    
    Learners{end+1} = curr_tr;
    
    final_hyp = final_hyp + step_out .* Alpha;    
  end
  
  distr = exp(- 1 * (Labels .* final_hyp));
  Z = sum(distr);
  distr = distr / Z;  

end
