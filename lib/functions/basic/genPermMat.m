function P = genPermMat(n, i, j)
%GENPERMMAT Generate a permutation matrix with size n and permutes ith
%and jth.
P = eye(n);
P(i, i) = 0;
P(j, j) = 0;
P(i, j) = 1;
P(j, i) = 1;
end
