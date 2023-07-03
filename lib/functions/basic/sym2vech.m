function mh = sym2vech(M)
%SYM2VECH Coverting a symmetric matrix into corresponding half vecor
issym = (M == M');
[a,~] = size(M); 
mh = zeros(1/2*(a*(a+1)),1);
if issym
    ivh = 1;
    for i = 1:a
        for j = i:a
            mh(ivh) = M(i,j);
            ivh = ivh + 1;
        end
    end
end
