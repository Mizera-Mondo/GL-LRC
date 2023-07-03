function n0 = norm0(M)
%NORM0 此处显示有关此函数的摘要
%   此处显示详细说明
M(abs(M) >= 1e-8) = 1;
M(abs(M) <= 1e-1) = 0;
n0 = sum(M, 'all');
end
