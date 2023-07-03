function dV = directedVariation(Adjacency, Signal)
%directedVariation Calculate directed variation with given adjacency matrix
%and signal. This variation metric is designed for directed graphs.
[n, m] = size(Signal);
if m > n
    Signal = Signal';
    n = m;
end
sigMat = kron(Signal, ones(1, n));
sigMat = sigMat - sigMat';
sigMat(sigMat < 0) = 0;
dV = sum(sum(Adjacency.*sigMat));
end
