% Add current paths
currentPath = pwd;
addpath(genpath(currentPath));
% Initialization
nodeNum = 30;
usedEigNum = 25;
signalLength = 100;
noiseCov = 0;
rPertubation = 0;
threA = 1e-3;
% Signal Generation
[Y, A, R] = genRandomSignal(nodeNum, usedEigNum, signalLength, noiseCov, rPertubation);

L = diag(sum(A)) - A;
B = zeros(signalLength);
B(1:end - 1, 2:end) = eye(signalLength - 1);
D = @(X) X - R*X*B;
alpha = 0.1;
beta = 5;

% For Debug: The Target Function
targ1 = @(X, L) alpha*trace((D(X))'*L*D(X));
targ2 = @(X, L) beta*norm(L, "fro")^2;
Ar = rand(nodeNum);
Ar = Ar - diag(diag(Ar));
Ar = Ar + Ar';
Ar = Ar./sum(Ar, "all")*nodeNum;
Lr = diag(sum(Ar)) - Ar;

% Estimation
[X, Lest] = GL_LRC(Y, R, usedEigNum, alpha = alpha, beta = beta, LowRankEst = false);

DX = D(X);
M = genM(DX);


% Results
errLap = norm(Lest - L, 'fro')/norm(L, 'fro');
Aest = diag(diag(Lest)) - Lest;
disp("============================================");
disp("Estimation finished. NMSE of Laplacian: " + num2str(100*errLap) + "%");
[a, r, p, fM] = classifierPerformance(A > threA, Aest > threA);
disp("Accuracy  Recall   Precision   f-Measure");
disp(num2str([a, r, p, fM]));

disp("Loss of groundtruth: " + num2str(targ1(X, L)) + ", " + num2str(targ2(X, L)));
disp("Loss of estimated: " + num2str(targ1(X, Lest)) + ", " + num2str(targ2(X, Lest)));
disp("Loss of random: " + num2str(targ1(X, Lr)) + ", " + num2str(targ2(X, Lr)));
% Visualized Results
% close all;
figure;
subplot(2, 2, 1)
imagesc(L); colorbar; title('Ground Truth');
subplot(2, 2, 2)
imagesc(Lest); colorbar; title('Estimated');
subplot(2, 2, 3)
[~, S, ~] = svd(Y); imagesc(S(:, 1:nodeNum)); colorbar; title("Singular Values of Sampled Signals");
subplot(2, 2, 4)
[~, S, ~] = svd(X); imagesc(S(:, 1:nodeNum)); colorbar; title("Singular Values of Denoised Signals");
