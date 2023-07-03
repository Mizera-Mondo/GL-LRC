function [eigVec, eigValue] = CMDS(D)
%CMDS Classical Multidimensional Scaling
% D: Distance matrix
[n, ~] = size(D);
J = eye(n) - 1/n*ones(n);
B = -0.5*J*D.^2*J;
[eigVec, eigValue] = eig(B);
[eigVec, eigValue] = sortEigen(eigVec, eigValue, 'descend');
end
