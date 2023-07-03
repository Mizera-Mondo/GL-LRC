function M = selectMaxKRows(M, k)
%SELECTMAXKROWS
    m = sum(M.^2, 2);
    [~, index] = sort(m, 'descend');
    indexToClear = setdiff(index, index(1:k));
    M(indexToClear, :) = 0;
end
