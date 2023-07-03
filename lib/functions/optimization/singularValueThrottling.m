function M = singularValueThrottling(M, k)
%SINGULARVALUETHROTTLING Keep only the biggest k singular values and
%vectors.
%   A more 'harder' limiter compared with singular value threshold.
[U, Z, V] = svd(M);
M = U(:, 1:k)*Z(1:k, 1:k)*(V(:, 1:k))';
end

