function [Y, yShift] = spectralShift(A, y, i)
%SPECTRALSHIFT 此处显示有关此函数的摘要
%   此处显示详细说明
[N, ~] = size(A);
L = diag(sum(A)) - A;
[vec, ~] = eig(L);
Y = vec'*y;
yShift = zeros(N, 1);
for k = 1:N
    for n = 1:N
        yShift(k) = yShift(k) + y(n)*vec(n, i)*vec(n, k);
    end
end
end
