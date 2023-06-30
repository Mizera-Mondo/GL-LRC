function X = SolveSubX(R, Y, B, L, alpha, options)
%SOLVESUBX solve 1/2||D(Y) - Z||_F^2 + alpha*tr(Z'*L*Z)
% s.t. Z = D(X) = X - RXB
% rank(X) <= k
arguments
    R, Y, B, L, alpha double
    options.rho = 1;
    options.tol = 1e-3
    options.debug = false
end
tol = options.tol;
rho = options.rho;

X = Y;
Z = D(X);
Phi = X;
Theta = Z - (X + R*X*B);
Xi = X - Phi;
D = @(X) X - R*X*B;

isConverge = false;
isMaxIter = false;
while ~isConverge && ~isMaxIter
    X_old = X;
    % Update of X
    "TODO: SOLVE X USE LYAP
    % Update of Z ?
    
    % Update of Phi
    Phi = singularValueThrottling(X + Xi/rho);
    % Update of Theta
    Theta = Theta + rho*(Z - (X - R*X*B));
    % Update of Xi
    Xi = Xi + rho*(X - Phi);
    % Terminal Condition Check
    primalRes = norm(Phi - X, 'fro')/norm(X, 'fro');
    dualRes = norm(X_old - X, 'fro')/norm(X_old, 'fro');
    isConverge = primalRes <= tol && dualRes <= tol;
end

end

