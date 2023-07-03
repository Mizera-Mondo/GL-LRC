function Md = dupmat(k)
%DUPMAT Generate a Duplication Matrix for a half vector with 1/2*(k^2+k) elements, mapping it into a
%full vector with k^2 elements representing the vectorized form of a symmetric matrix.
Md = zeros(k*k, 1/2*(k*(k+1)));
ivh = 1; % index of half vector
for i = 1:k % column of square matrix
    Md((k+1)*i - k, ivh) = 1; % mapping on-diagonal elements 
    ivh = ivh + 1;
    for j = i+1:k % row of square matrix
        Md(k*(i-1) + j, ivh) = 1;
        Md(k*(j-1) + i, ivh) = 1;
        ivh = ivh +1;
    end
end
end
