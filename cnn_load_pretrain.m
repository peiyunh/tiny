function net = cnn_load_pretrain(net, prepath)

% convert pretrained network to DagNN (easy indexing) 
prenet_ = load(prepath);
if isfield(prenet_, 'net')
    prenet_ = prenet_.net;
end

if isfield(prenet_, 'params')
    prenet = dagnn.DagNN.loadobj(prenet_);
else
    prenet = dagnn.DagNN.fromSimpleNN(prenet_);
end

clear prenet_;

% same canonical param name
if isempty(net)
    net = prenet;
else
    for i = 1:numel(net.params) 
        idx = prenet.getParamIndex(net.params(i).name); 
        if ~isnan(idx)
            net.params(i).value = prenet.params(idx).value; 
        end
    end
end

if isempty(net.getLayerIndex('drop6'))
    net.addLayer('drop6', dagnn.DropOut('rate', 0.5), 'fc6x', 'fc6xd');
    net.setLayerInputs('fc7', {'fc6xd'});
end

if isempty(net.getLayerIndex('drop7'))
    net.addLayer('drop7', dagnn.DropOut('rate', 0.5), 'fc7x', 'fc7xd');
    net.setLayerInputs('score_fr', {'fc7xd'});
end

% remove average image 
net.meta.normalization.averageImage = []; 

% NOTE Reshape multipliers and biases in BN to be vectors instead of
% 1x1xK matrices. Otherwise, there will be errors in cnn_train_dag.m
for i = 1:numel(net.layers)
    if isa(net.layers(i).block, 'dagnn.BatchNorm')
        midx = net.getParamIndex(net.layers(i).params{1}); % multiplier
        bidx = net.getParamIndex(net.layers(i).params{2}); % bias
        % vectorize 
        net.params(midx).value = net.params(midx).value(:);
        net.params(bidx).value = net.params(bidx).value(:);
    end
end