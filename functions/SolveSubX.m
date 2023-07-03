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

D = @(X) X - R*X*B;

X = Y;
Z = D(X);
Phi = X;
Theta = Z - (X + R*X*B);
Xi = X - Phi;

I = eye(size(L))
DY = D(Y);
Kz = inv(2*alpha*L + (1 + rho)*L);

isConverge = false;
isMaxIter = false;
while ~isConverge && ~isMaxIter
    X_old = X;
    
    % Update of X
    % The update of X is divided into sub-problems w.r.t. X's rows.
    for i = 1:n
        z = Z(i, :) + 1/rho*Phi(i, :);
        phi = Phi(i, :) - 1/rho*Xi(i, :);
        r = R(i, i);
        X(i, :) = (2*I + r^2*B*B' - r*(B + B'))\((I - r*B)*z + phi);
    end

    % Update of Z
    Z = Kz*(DY + rho*D(X) - Theta);

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

