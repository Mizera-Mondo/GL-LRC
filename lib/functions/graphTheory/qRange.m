function qMax = qRange(A, tol)
%QRANGE Calculate the upper bound of q for Hermitian Laplacian.
% Reference: Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian
[N, ~] = size(A);
d = sum(sum(abs(A)))/N;
qMax = acos(1 - 2*tol/d)/(2*pi);
end
