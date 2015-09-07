function data = denormalize_data(data, params, index)
if nargin < 3
    index = 1:size(data,1);
end
ddata= params.mdata(index) - params.ndata(index);
if size(data,1) > length(index)
    data = data(index,:);
end
data = bsxfun(@times, data+1, ddata/2);
data = bsxfun(@plus, data, params.ndata(index));