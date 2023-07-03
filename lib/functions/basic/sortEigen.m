function [vecSorted, valSorted] = sortEigen(vec, val, direction)
%SORTEIGEN sort the eigenvectors and eigenvalues by ascend or descend
%order.
%   此处显示详细说明
[~, index] = sort(diag(val), direction);
valSorted = val(index, index);
vecSorted = vec(:, index);
end
