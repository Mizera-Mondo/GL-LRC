function X = PCA_power_iteration(D, k)
%PCA_POWER_ITERATION Calculate PCA of data matrix D \in R^{n*p}, output the first k
%components X \in R^{k*p}.
%   Components will be found one by one with power iteration instead of
%   using SVD. 
if ~isint(k)
    error('k = %g is required to be an integer.', k);
end
[n, p] = size(D);
X = zeros(k, p);

%% Iteration
end

