function hLaplacian = hermitianLaplacian(A, q, normalizeMark)
% Genertate an Hermitian Laplacian with given adjacency matrix A and
% rotation constant q.
% Reference: Graph Signal Processing for Directed Graphs based on the Hermitian Laplacian
As = (A + A')/2;
Gamma_q = exp(1i*2*pi*q*(A - A'));
Ds = diag(sum(As));
hLaplacian = Ds - Gamma_q.*As;
if normalizeMark
    normalizedDs = diag(diag(Ds).^(-1/2));
    hLaplacian = normalizedDs*hLaplacian*normalizedDs;
end
end
