function G = laplacianProjection(M)
%LAPLACIANPROJECTION Project the input matrix so as to get a nearest valid Laplacian.
%   Input: M

% Initialization

[N, ~] = size(M);
if mean(abs(M), 'all') > 1e10
    warning(['The input matrix has too large elements.'  newline  'This may lead to numerical problems.']);
end

G = M;
K = M;

Phi = zeros(N);
rho = 0.3;
m = mat2vec(M);

% Quadratic constriants
A = [];
b = [];
Aeq = [];
beq = [];

% Gij = Gji <= 0
for i = 1:N
    for j = 1:i - 1

        T = zeros(N);
        T(i, j) = 1;
        A = [A, mat2vec(T)];
        b = [b, 0];

        T(j, i) = -1;
        Aeq = [Aeq, mat2vec(T)];
        beq = [beq, 0];

    end
end

O = zeros(N);
% G1 = 0
for i = 1:N
    T = zeros(N);
    T(i, :) = ones(1, N);
    Aeq = [Aeq, mat2vec(T)];
    beq = [beq, 0];
    O = O + T;
end
% 
% % trace(G) = N
Aeq = [Aeq, mat2vec(eye(N))];
beq = [beq, N];

A = A';
b = b';
Aeq = Aeq';
beq = beq';


% Technical varbs
iter = 1;
maxIter = 100;
isAdmmConverge = false;
isMaxIter = false;

while ~isAdmmConverge && ~isMaxIter
    G_old = G;
    K_old = K;
    % Non-SPD part w.r.t G, using quadratic programming solver
    g = mat2vec(G);
    k = mat2vec(K);
    u = mat2vec(Phi);
    H = (rho + 1)*eye(length(g));
    f = (u - m - rho*k);
    g = quadprog(H, f, A, b, Aeq, beq, [], [], g, optimoptions(@quadprog, 'Display', 'off'));
    G = vec2squ(g);
    % SPD part w.r.t K, using 1-step SPD projection
    U = G + 1/rho*Phi;
    K = nearestSPD(U);
    
    % Update Phi
    Phi = Phi + rho*(G - K);
    % Convergence Check
    isAdmmConverge = norm(G - G_old, 'fro')/norm(G_old, 'fro') < 1e-3 && ...
        norm(K - K_old, 'fro')/norm(K_old, 'fro') < 1e-3;
    isMaxIter = iter >= maxIter;
    iter = iter + 1;
end
end
