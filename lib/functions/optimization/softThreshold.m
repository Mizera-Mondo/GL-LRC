function x_t = softThreshold(X, t)
%SOFTTHRESHOLD 此处显示有关此函数的摘要
%   此处显示详细说明
x_t = sign(X).*(((abs(X) - t)>0).*(abs(X) - t));
end
