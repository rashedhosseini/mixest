function params = normalize_param(data)
params.mdata = max(data,[],2);
params.ndata = min(data,[],2);