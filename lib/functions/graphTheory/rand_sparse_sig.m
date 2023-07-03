function s = rand_sparse_sig(nv, len ,spa, val)
%RAND_SPARSE_SIG 输出一个稀疏的信号矩阵
% nv: 顶点数量
% len: 采样数量
% spa: 稀疏度, 限制为0-1
% val: 信号值
%%
s = zeros(nv, len);
coord = randperm(nv*len, round(nv*len*spa));
for i = 1:length(coord)
    col = floor(coord(i)/nv);
    row = coord(i) - col*nv;
    if row ~= 0
        s(row, col+1) = val;        % Normal situation
    else
        s(nv, col) = val;              % Edge of column
    end
end
