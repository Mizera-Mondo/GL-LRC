% Add current paths
currentPath = pwd;
addpath(genpath(currentPath));
% Initialization
nodeNum = 10;
usedEigNum = 9;
signalLength = 1000;
noiseCov = 0.1;
rPertubation = 0.01;

% Signal Generation
[Y, A, R] = genRandomSignal(nodeNum, usedEigNum, signalLength, noiseCov, rPertubation);

L = diag(sum(A)) - A;
B = zeros(signalLength);
B(2:end, 2:end) = eye(signalLength - 1);
D = @(X) X - R*X*B;
alpha = 0.1;
beta = 225;

% Estimation
[X, Lest] = GL_LRT(Y, R, usedEigNum, alpha = alpha, beta = beta);

% Results
errLap = norm(Lest - L, 'fro')/norm(L, 'fro');
disp("============================================");
disp("Estimation finished. NMSE of Laplacian: " + num2str(100*errLap) + "%");
close all;
figure; imagesc(L); colorbar; title('Ground Truth');
figure; imagesc(Lest); colorbar; title('Estimated');
figure; [~, S, ~] = svd(Y); imagesc(S(:, 1:nodeNum)); title("Singular Values of Sampled Signals");
figure; [~, S, ~] = svd(X); imagesc(S(:, 1:nodeNum)); title("Singular Values of Denoised Signals");