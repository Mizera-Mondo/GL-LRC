function [X, L, A] = GL_LRT(Y, R, k, options)
%LR-DGI Solve 1/2||D(Y - X)||_F^2 + alpha*Tr{D(X)'*L*D(X)} +
%beta/2||L||_F^2

arguments
    Y, R, k double
    options.alpha = 0.1;
    options.beta = 0.1;
    options.graphRefineMethod = 'quadprog';
    options.LowRankEst = true;
    options.debug = false;
end
alpha = options.alpha;
beta = options.beta;
debug = options.debug;

% k is the maximum rank of estimated signal matrix X
X = Y;

% Random vaild initial value of A
[n, T] = size(Y);
B = [zeros(T - 1, 1) eye(T - 1); ...
    zeros(1, T)];
D = @(X) X - R*X*B;
A = rand(n, n);
A = A - diag(diag(A));
A = A + A';
A = n*A./sum(A, 'all');
L = diag(sum(A)) - A;

tol = 1e-2;
iter = 1;
maxIter = 1000;
isConverge = false;
isMaxIter = false;

while ~isConverge && ~isMaxIter

    L_old = L;
    X_old = X;
    % Optimizing L a.k.a. A
    DX = D(X);
    M1 = DX*DX';
    M2 = repmat(diag(M1), 1, n);
    M = M2 + M2' - 2*M1;
    % M = M./T;

    tic
    A = SolveSubA(M, alpha, beta);
    toc

    L = diag(sum(A)) - A;

    % Optimizing X
    % X = solveSubX(Y, L, R, B, alpha, k);
    if options.LowRankEst
        tic
        X = updateX_SVD(X, D(Y), L, R, B, alpha, k, solver = 'GPM');
        toc
    end
    
    isConverge = norm(L_old - L, 'fro')/norm(L_old, 'fro') < tol ...
            && norm(X_old - X, 'fro')/norm(X_old, 'fro') < tol;
    isMaxIter = iter >= maxIter;
    iter = iter + 1;
end
end


