nodeNum = 30;
sp = zeros(nodeNum, length(0.2:0.1:0.9));
i = 1;
for thre = 0.2:0.1:0.9
    for j = 1:100
        [~, L] = genNormalRGG(30, thre = thre);
        [~, spct] = eig(L);
        sp(:, i) = sp(:, i) + diag(spct);
    end
    sp(:, i) = sp(:, i)./100;
    i = i + 1;
end
close all;
plot(1:nodeNum, sp); legend; grid;
