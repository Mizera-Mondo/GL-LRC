% Add current paths
currentPath = pwd;
addpath(genpath(currentPath));
% Initialization
nodeNum = 10;
usedEigNum = 9;
signalLength = 1000;
noiseCov = 0.1;
rPertubation = 0.01;
threA = 1e-3;
% Signal Generation
[Y, A, R] = genRandomSignal(nodeNum, usedEigNum, signalLength, noiseCov, rPertubation);

L = diag(sum(A)) - A;
B = zeros(signalLength);
B(2:end, 2:end) = eye(signalLength - 1);
D = @(X) X - R*X*B;
alpha = 0.1;
beta = 200;

% Estimation
[X, Lest] = GL_LRT(Y, R, usedEigNum, alpha = alpha, beta = beta);

% Results
errLap = norm(Lest - L, 'fro')/norm(L, 'fro');
Aest = diag(diag(Lest)) - Lest;
disp("============================================");
disp("Estimation finished. NMSE of Laplacian: " + num2str(100*errLap) + "%");
[a, r, p, fM] = classifierPerformance(A > threA, Aest > threA);
disp("Accuracy  Recall   Precision   f-Measure");
disp(num2str([a, r, p, fM]));
% Visualized Results
close all;
figure;
subplot(2, 2, 1)
imagesc(L); colorbar; title('Ground Truth');
subplot(2, 2, 2)
imagesc(Lest); colorbar; title('Estimated');
subplot(2, 2, 3)
[~, S, ~] = svd(Y); imagesc(S(:, 1:nodeNum)); title("Singular Values of Sampled Signals");
subplot(2, 2, 4)
[~, S, ~] = svd(X); imagesc(S(:, 1:nodeNum)); title("Singular Values of Denoised Signals");
