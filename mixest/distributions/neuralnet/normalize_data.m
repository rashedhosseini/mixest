function data = normalize_data(data, params, index)
if nargin < 3
    index = 1:size(data,1);
end
data = bsxfun(@minus, data(index,:), params.ndata(index));
ddata = params.mdata(index) - params.ndata(index);
data = bsxfun(@rdivide, data, ddata)*2-1;