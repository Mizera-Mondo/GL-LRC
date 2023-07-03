function M = vec2squ(v)
%VEC2SQU 此处显示有关此函数的摘要
%   此处显示详细说明
[a,b] = size(v);
sa = floor(sqrt(a));
if b == 1 && a == sa^2
    M = reshape(v,sa,sa);
else
    M = 0;
end
end
