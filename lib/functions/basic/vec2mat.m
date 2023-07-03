function M = vec2mat(v, r)
%VEC2MAT Reshape a vector to a matrix.
%   r is the number of rows of the new matrix and the number of column will
%   be automatically detected.
[a,b] = size(v);

if b == 1 && mod(a, r) == 0
    M = reshape(v, r, a/r);
else
    M = 0;
    if b ~= 1
        error('Input must be a vector.');
    else 
        error('Could not construct a %d-row-matrix with input vector.', r);
    end
end
end
