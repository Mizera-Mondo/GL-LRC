function X = SolveSubX(Y, R, B, L, alpha, k, options)
%SOLVESUBX solve 1/2||D(Y) - Z||_F^2 + alpha*tr(Z'*L*Z)
% s.t. Z = D(X) = X - RXB
% rank(X) <= k
arguments
    Y, R, B, L, alpha, k double
    options.rho = 100
    options.tol = 1e-3
    options.maxIter = 2000
    options.debug = false
end
tol = options.tol;
rho = options.rho;

[n, m] = size(Y);

D = @(X) X - R*X*B;

X = Y;
Z = D(X);
Phi = X;
Theta = Z - (X + R*X*B);
Xi = X - Phi;

I = eye(n);
Ix = eye(m);
DY = D(Y);


isConverge = false;
isMaxIter = false;
iter = 1;
while ~isConverge && ~isMaxIter
    X_old = X;
    % Update of X
    % The update of X is divided into sub-problems w.r.t. X's rows.
    for i = 1:n
        z = Z(i, :) + 1/rho*Theta(i, :);
        z = z';
        phi = Phi(i, :) - 1/rho*Xi(i, :);
        phi = phi';
        r = R(i, i);
        x = (2*Ix + r^2*B*B' - r*(B + B'))\((Ix - r*B)*z + phi);
        X(i, :) = x';
    end
    % Update of Z
    Z = (2*alpha*L + (1 + rho)*I)\(DY + rho*D(X) - Theta);
    % Update of Phi
    Phi = singularValueThrottling(X + Xi/rho, k);
    % Update of Theta
    Theta = Theta + rho*(Z - (X - R*X*B));
    % Update of Xi
    Xi = Xi + rho*(X - Phi);
    % Terminal Condition Check
    primalRes = norm(Phi - X, 'fro')/norm(X, 'fro');
    dualRes = norm(X_old - X, 'fro')/norm(X_old, 'fro');
    isConverge = primalRes <= tol && dualRes <= tol;
    isMaxIter = iter >= options.maxIter;
    iter = iter + 1;
end

end


