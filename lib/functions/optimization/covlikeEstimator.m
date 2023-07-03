function A = covlikeEstimator(Sig, model)
%COVLIKEESTIMATOR Estimate graph topology with certain covariance pattern.
%   Sig: Sampled signals, model: estimator model
[n, l] = size(Sig);
A = zeros(n);

R0 = Sig*Sig'/l;
% TODO: FILL IN THE ESTIMATOR
if model == "granger"

elseif model == "1lag"
elseif model == "residual"
elseif model == "nig"
end

end
