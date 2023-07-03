function [X, seqX] = adaDelta(Gd, X, epsilon, rho)
%ADADELTA AdaDelta algorithm for unconstraint convex optimization.
%  Tg: Target function
%  Gd: Gradient of target function
%  X: Starting point of variable
%  epsilon: Tweaking parametre, rec. 1e-4
%  rho: Forgetting rate, rec. 0.95
eGd = zeros(size(X));
eDx = eGd;
seqX = X;
maxIter = 5000;
tol = 1e-4;
iter = 0;
isConverge = false;
isMaxIterReach = false;
while ~isConverge && ~isMaxIterReach
    gradD = Gd(X);
    eGd = rho*eGd + (1 - rho)*(gradD.^2);
    rmsGd = (eGd + epsilon).^(1/2);
    rmsDx = (eDx + epsilon).^(1/2);
    dX = -(rmsDx./rmsGd).*gradD;
    X = X + dX;
    eDx = rho*eDx + (1 - rho)*(dX.^2);

    % Convergence check
    isConverge = norm(dX, "fro")/(norm(X, "fro") + 1e-10) <= tol;
    
    iter = iter + 1;
    isMaxIterReach = iter >= maxIter;
    if floor(iter/1000)*1000 == iter
        disp(iter);
        %disp(X);
    end
    seqX = [seqX X];
end
%optValue = Tg(X);

end
