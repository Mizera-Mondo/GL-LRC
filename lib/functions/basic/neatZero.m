function X = neatZero(X, options)
%NEATZERO Compress any small numbers in X whose abstract less than th to 0.
% th: the threshold. optional, default = 1e-7.
arguments
    X double
    options.th = 1e-7
end
X(abs(X) < options.th) = 0;
end

