%% |nnfactory|
% Construct a neural network distribution structure
%
% *Syntax*
%
%   D = nnfactory(datadim, num)
%
% *Description*
%
% |D = nnfactory(num)| returns a structure representing a nnet with
%  |num| number of hidden units and |datadim| dimensionality of input 
%
% *Distribution Parameters*
%
% * *|W|* (|datadim-by-num| matrix) : Weights of the input layer.
% * *|b|* (|datadim-by-1| vector): Biases of the input layer 
% * *|h|* (|num-by-1| vector) : The weights of the output layer.
% * *|s|* (scalar): the bias of the output layer.
%
% *Probability Density Function*
%
% The distribution has the following density:
% 
% $$ f(y|x)=
% (2\pi)^{-\frac{n}{2}} \exp\left(-\frac{1}{2}(y-\mu(x))^2 \right) $$
%
% where $n$ is the data space dimensions, $\mu(x)$ is the output of the 
% neural network computed
% $$ \mu(x) = h^T * tanh(W*x+b) + s $$
% 

%
% Contributors:
%  Rashed Hosseini
%
% Change log: 
%

function D = nnfactory(datadim, num)

%% |name|
% See <doc_distribution_common.html#1 distribution structure common members>.

    D.name = @() 'nnet';

%%

    assert(datadim >= 1, 'datadim must be an integer larger than or equal to 1.');

%% |M|
% See <doc_distribution_common.html#2 distribution structure common members>.

    WM = euclideanfactory(datadim, num);
    bM = euclideanfactory(num);
    hM = euclideanfactory(num);
    sM = euclideanfactory(1);
    D.M = productmanifold(struct('h', hM, 's', sM, 'b', bM,'W', WM));

%% |dim|
% See <doc_distribution_common.html#3 distribution structure common members>.

    D.dim = @() (datadim+2)*num + 1; % parameter space dimensions

%% |datadim|
% See <doc_distribution_common.html#4 distribution structure common members>.

    D.datadim = @() datadim; % data space dimensions

%%
    function store = intermediate_func(theta, data, store)
        
        if ~isfield(store, 'Wxb')
            store.Wxb = bsxfun(@plus, theta.W.' * data(1:end-1,:), theta.b);
            store.tWxb = tanh(store.Wxb);
            store.htWxb = theta.h.' * store.tWxb + theta.s;
            store.e = store.htWxb - data(end,:);
        end
        
    end

%% |ll|
% See <doc_distribution_common.html#5 distribution structure common members>.

    D.ll = @ll;
    function [ll, store] = ll(theta, data, store)
        
        if nargin < 3
            store = struct;
        end
        
        [llvec, store] = D.llvec(theta, data, store);

        % Calculate the log-likelihood
        ll = sum(llvec);
        
    end

%% |llvec|
% See <doc_distribution_common.html#6 distribution structure common members>.

    D.llvec = @llvec;
    function [llvec, store] = llvec(theta, data, store)
        
        if nargin < 3
            store = struct;
        end
        
        data = mxe_readdata(data);
        
        weight = data.weight;
        data = data.data;
        
        store = intermediate_func(theta, data, store);
        
        llvec = - 0.5 * store.e.^2;            
        if ~isempty(weight)
            llvec = weight .* llvec;
        end
    end

%% |llgrad|
% See <doc_distribution_common.html#7 distribution structure common members>.

    D.llgrad = @llgrad;
    function [dll, store] = llgrad(theta, data, store)
        
        if nargin < 3
            store = struct;
        end
        
        data = mxe_readdata(data);
        
        weight = data.weight;
        data = data.data;
        
        store = intermediate_func(theta, data, store);

        % gradient with respect to sigma
        if isempty(weight)
            err = store.e;
        else
            err = store.e .* weight;
        end
        
        dll.h = - sum(bsxfun(@times,store.tWxb, err), 2);
        dll.s = - sum(err);
        cWxb = 1 - store.tWxb.^2;
        ercWxb = bsxfun(@times, err, cWxb);
        dll.b = - theta.h .* sum(ercWxb,2);
        dll.W = - bsxfun(@times, theta.h.', data(1:end-1,:) * ercWxb.');
        %theta.h.' * sum(bsxfun(@times, cWxb.*er, data(1:end-1,:)),2).';
        
    end

%% |llgraddata|
% See <doc_distribution_common.html#8 distribution structure common members>.

        
%% |cdf|
% See <doc_distribution_common.html#9 distribution structure common members>.

%% |pdf|
% See <doc_distribution_common.html#10 distribution structure common members>.

%% |sample|
% See <doc_distribution_common.html#11 distribution structure common members>.

%% |randparam|
% See <doc_distribution_common.html#12 distribution structure common members>.

%% |init|
% See <doc_distribution_common.html#13 distribution structure common members>.

    D.init = @init;
    function theta = init(data, varargin)
        
        theta = estimatedefault(data);
    end

%% |estimatedefault|
% Default estimation function for multi-variate normal distribution. This
% function implements the maximum likelihood method.

%% |penalizerparam|
% See <doc_distribution_common.html#15 distribution structure common members>.
%

%% |penalizercost|
% See <doc_distribution_common.html#16 distribution structure common members>.


%% |penalizergrad|
% See <doc_distribution_common.html#17 distribution structure common members>.


%% |sumparam|
% See <doc_distribution_common.html#18 distribution structure common members>.

    D.sumparam = @sumparam;
    function theta = sumparam(theta1, theta2)
        theta.W = theta1.W + theta2.W;
        theta.b = theta1.b + theta2.b;
        theta.h = theta1.h + theta2.h;
        theta.s = theta1.s + theta2.s;
    end

%% |scaleparam|
% See <doc_distribution_common.html#19 distribution structure common members>.

    D.scaleparam = @scaleparam;
    function theta = scaleparam(scalar, theta)
        theta.W = scalar * theta.W;
        theta.b = scalar * theta.b;
        theta.h = scalar * theta.h;
        theta.s = scalar * theta.s;
    end

%% |sumgrad|
% See <doc_distribution_common.html#20 distribution structure common members>.

%% |scalegrad|
% See <doc_distribution_common.html#21 distribution structure common members>.

%% |entropy|
% See <doc_distribution_common.html#22 distribution structure common members>.

%% |kl|
% See <doc_distribution_common.html#23 distribution structure common members>.

%% |AICc|
% See <doc_distribution_common.html#24 distribution structure common members>.

%% |BIC|
% See <doc_distribution_common.html#25 distribution structure common members>.

%% |display|
% See <doc_distribution_common.html#26 distribution structure common members>.

    D.display = @display;
    function str = display(theta)
        str = '';
        str = [str, sprintf('Input Weights (%d-by-%d): %s\n', size(theta.W,1), size(theta.W,2), mat2str(theta.W, 4))];
        str = [str, sprintf('Input Bias (%d-by-1): %s\n', size(theta.b,1), mat2str(theta.b, 4))];
        str = [str, sprintf('Output Weights (%d-by-1): %s\n', size(theta.h,1), mat2str(theta.h, 4))];
        str = [str, sprintf('Output Bias (1-by-1): %s\n', mat2str(theta.s, 4))];
        
        if nargout == 0
            str = [sprintf('%s distribution parameters:\n', D.name()), str];
        end
    end

%% |visualize|
%

%%

    D = mxe_addsharedfields(D);
end
