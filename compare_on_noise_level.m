% Add current paths
currentPath = pwd;
addpath(genpath(currentPath));
% Initialization
nodeNum = 15;
signalLength = 100;
noiseCov = 0;
rPertubation = 0;
threA = 1e-3;
usedEigNum = 5;

kMax = 10;

alpha = 0.1;
beta = 10;

serie = 0:0.1:5;
perf = zeros(length(serie), 5);
perf(:, 1) = serie;


for k = 1:kMax
    if mod(k, 10) == 0
        disp("current Iteration: " + num2str(k));
    end
    % [A, L] = genRandomGraph(nodeNum);
    [A, L] = genRandomGraph(nodeNum);
    i = 1;
    for noiseCov = serie
            % Signal Generation
    
            [Y, R] = genRandomSignal(L, usedEigNum, signalLength, noiseCov, rPertubation);
            
            % Estimation

            B = zeros(signalLength);
            B(1:end - 1, 2:end) = eye(signalLength - 1);
            D = @(X) X - R*X*B;
            Lest = GL_logdet(D(Y));
            % [X, Lest] = GL_SigRep(D(Y), 0.1, 5, 1000);
            % [X, Lest] = GL_LRSS(Y, alpha = alpha, beta = 5, gamma = 2);
            % [X, Lest] = GL_LRSS(Y, alpha = alpha, beta = beta, LowRankEst = false);
            Aest = diag(diag(Lest)) - Lest;
            
            [a, r, p, fM] = classifierPerformance(A > threA, Aest > threA);
            perf(i, 2:5) = perf(i, 2:5) + [a, r, p, fM];
            i = i + 1;
    end
end
perf(:, 2:5) = perf(:, 2:5)./kMax;
plot(perf(:, 1), perf(:, 2:5), LineWidth=2);
legend("Acc", "Pre", "Rec", "f-M");
grid on
save("etc/noise_level_logdet.mat");