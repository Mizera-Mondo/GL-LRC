function n = nuclearNorm(M)
%NUCLEARNORM Calculate the nuclear norm n of matrix M.
% n = trace((M'*M)^(1/2));
[~, S, ~] = svd(M);
n = sum(diag(S));
end

