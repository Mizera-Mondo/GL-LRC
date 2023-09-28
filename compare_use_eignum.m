% Add current paths
currentPath = pwd;
addpath(genpath(currentPath));
% Initialization
nodeNum = 30;
signalLength = 100;
noiseCov = 0;
rPertubation = 0;
threA = 1e-3;

kMax = 30;

alpha = 0.1;
beta = 5;

perf = zeros(length(25:25), 5);
perf(:, 1) = 25:25;


for k = 1:kMax
    disp("current Iteration: " + num2str(k));
    [A, L] = genRandomGraph(nodeNum);
    % [A, L] = genNormalRGG(nodeNum, thre=0.7);
    i = 1;
    for usedEigNum = (perf(:, 1))'
            % Signal Generation
    
            [Y, R] = genRandomSignal(L, usedEigNum, signalLength, noiseCov, rPertubation);
            
            % Estimation
            % [X, Lest] = GL_LRC(Y, R, usedEigNum, alpha = alpha, beta = beta, LowRankEst = false);
            [X, Lest] = GL_LRSS(Y, alpha = alpha, beta = beta, LowRankEst = false);
            Aest = diag(diag(Lest)) - Lest;
            
            [a, r, p, fM] = classifierPerformance(A > threA, Aest > threA);
            perf(i, 2:5) = perf(i, 2:5) + [a, r, p, fM];
            i = i + 1;
    end
end
perf(:, 2:5) = perf(:, 2:5)./kMax;
save("etc/perfOverRGGsEigNum_with0-1_noise.mat");
plot(perf(:, 1), perf(:, 2:5), LineWidth=2);
legend("Acc", "Pre", "Rec", "f-M");
grid on
