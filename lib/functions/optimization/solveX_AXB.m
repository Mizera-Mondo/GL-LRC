function X = solveX_AXB(A, B, C)
%SOLVEX_AXB Sovling matrix equation with form of X - AXB = C.
% Ref: On solutions of the matrix equations X− AXB = C
[m, ~] = size(A);
[~, n] = size(B);
coffF = poly(A);
coffH = coffF(end:-1:1);
X = zeros(m, n);
for k = 1:m
    for s = 1:k
        X = X + coffH(k + 1)*A^(k - s)*C*B^(m - s);
    end
end
X = X/(polyvalm(coffH, B));
end
