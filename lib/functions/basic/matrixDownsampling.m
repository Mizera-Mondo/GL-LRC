function mDownsampled = matrixDownsampling(M, sampleSizeA, sampleSizeB)
%MATRIXDOWNSAMPLING Using a sampleSize kernel to downsample input matrix M
[a, b] = size(M);
sa = ceil(a/sampleSizeA);
sb = ceil(b/sampleSizeB);
eleCount = sampleSizeA*sampleSizeB;
mExtend = zeros(sa*sampleSizeA, sb*sampleSizeB);
mExtend(1:a, 1:b) = M;
mDownsampled = zeros(sa, sb);
for i = 1:sa
    for j = 1:sb
        startPos = [(i - 1)*sampleSizeA + 1, (j - 1)*sampleSizeB + 1];
        endPos = startPos + [sampleSizeA, sampleSizeB] - 1;
        mDownsampled(i, j) = sum(sum(mExtend(startPos(1):endPos(1), startPos(2):endPos(2))))/eleCount;
    end
end
end
