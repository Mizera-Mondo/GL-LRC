function v = mat2vec(M)
%MAT2VEC 此处显示有关此函数的摘要
%   此处显示详细说明
[a,b] = size(M);
v = reshape(M,a*b,1);
end
