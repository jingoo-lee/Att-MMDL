function [kernelsize,layers] = experimental_layer_v3(kernelsize)


layers = dlnetwork;

r1 = 1;  % dec1_ups
r2 = 1;  % dec2_ups
r3 = 1;  % dec3_ups

numFilters = 32;

k   = max(1, round(kernelsize));        % e.g., 64
k2  = max(1, round(kernelsize/2));      % e.g., 32
k10 = max(1, round(kernelsize/8));     % e.g., 8

layers1 = [
    sequenceInputLayer(1, 'Normalization','zscore', 'Name','input_cnn')

    convolution1dLayer(k, numFilters, Padding="same", Name="s_conv1")
    batchNormalizationLayer(Name="s_bn1")
    swishLayer(Name="s_sw1")

    residualBlockLayer(numFilters,1,k,true,"s_res1")
    residualBlockLayer(numFilters*2,1,k2,true,"s_res2")
    residualBlockLayer(numFilters*4,1,k10,true,"s_res3")
    
    functionLayer(@(X) X,Formattable=true, Acceleratable=false, Name = "branch1")
    ];

layers = addLayers(layers,layers1);

layers2 = [
    featureInputLayer(4,'Normalization','zscore','Name','struct_in')

    fullyConnectedLayer(32,'Name','kv_fc1282')
    batchNormalizationLayer('Name','kv_bn2')
    swishLayer('Name','kv_feat2')

    fullyConnectedLayer(128,'Name','kv_fc128')
    batchNormalizationLayer('Name','kv_bn')
    swishLayer('Name','kv_feat')

    functionLayer(@(X) cbtocbt(X),Formattable=true, Acceleratable=true, Name = "featurelast1")
    ];
layers = addLayers(layers,layers2);

layers3 = [
    % additionLayer(2,"Name",'addition')
    functionLayer(@(x,y) broadcast_cat(x,y), 'Formattable', true, 'Acceleratable', false, 'Name', 'addition')
    % concatenationLayer(1,2,'Name','addition')
    
    functionLayer(@(X) upsample_repeat(X, r1), 'Name','dec1_ups','Formattable',true,'Acceleratable',false)
    convolution1dLayer(k10, numFilters*4, Padding="same", Name="dec1_conv")
    batchNormalizationLayer(Name="dec1_bn")
    swishLayer(Name="dec1_sw")

    functionLayer(@(X) upsample_repeat(X, r2), 'Name','dec2_ups','Formattable',true,'Acceleratable',false)
    convolution1dLayer(k2, numFilters*2, Padding="same", Name="dec2_conv")
    batchNormalizationLayer(Name="dec2_bn")
    swishLayer(Name="dec2_sw")

    functionLayer(@(X) upsample_repeat(X, r3), 'Name','dec3_ups','Formattable',true,'Acceleratable',false)
    convolution1dLayer(k, numFilters*1, Padding="same", Name="dec3_conv")
    batchNormalizationLayer(Name="dec3_bn")
    swishLayer(Name="dec3_sw")

    convolution1dLayer(1, 60, Padding="same", Name="y_out")
    ];
layers = addLayers(layers,layers3);

layers = connectLayers(layers,"branch1","addition/in1");
layers = connectLayers(layers,"featurelast1","addition/in2");

    function Y = cbtocbt(X)
        idxC = finddim(X,"C");
        idxB = finddim(X,"B");

        % sizeS = size(X,idxS);
        sizeC = size(X,idxC);

        if ~isempty(idxB)
            numChannels = sizeC;
            sizeB = size(X,idxB);

            X = reshape(X,[numChannels sizeB 1]);
            Y = dlarray(X,"CBT");
        end

    end

end

function layer = residualBlockLayer(numFilters, ~, kernelsize, includeSkipConvolution, name)
g = dlnetwork;
main = [
    functionLayer(@identity, Formattable=false, Acceleratable=true, Name="split")

    convolution1dLayer(kernelsize,numFilters,Padding="same",Name="conv1")
    batchNormalizationLayer(Name="bn1")
    swishLayer(Name="sw1")

    convolution1dLayer(kernelsize,numFilters,Padding="same",Name="conv2")
    batchNormalizationLayer(Name="bn2")

    additionLayer(2,Name="add")
    swishLayer(Name="out")
    ];
g = addLayers(g, main);

if includeSkipConvolution
    skip = [
        convolution1dLayer(1,numFilters,Padding="same",Name="skipConv")
        batchNormalizationLayer(Name="bnSkip")
        ];
    g = addLayers(g, skip);
    g = connectLayers(g, "split",  "skipConv");
    g = connectLayers(g, "bnSkip", "add/in2");
else
    g = connectLayers(g, "split", "add/in2");
end

layer = networkLayer(g, Name=name);
end

function Y = identity(X), Y = X; end

function Y = upsample_repeat(X, r)
% X: dlarray 'CBT' or raw [C B T]
% Y: [C B (r*T)] by repeat along time dim. If r=1, identity.
if isa(X,'dlarray')
    lbl = dims(X);
    Xn = stripdims(X);
else
    lbl = 'CBT';
    Xn = X;
end
Yd = repelem(Xn, 1, 1, r);  % repeat along time dimension
Y  = dlarray(Yd, lbl);
end

function Z = broadcast_cat(X_seq, X_static)

T = size(X_seq, 3);


X_static_expanded = repmat(X_static, 1, 1, T);


Z = cat(1, X_seq, X_static_expanded);
end