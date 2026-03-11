function [net] = createSeismicAttentionModel_vFinal2(kernelsize)

numFilters = 32;
kernelsize = kernelsize;

k   = max(1, round(kernelsize));        % e.g., 64
k2  = max(1, round(kernelsize/2));      % e.g., 32
k10 = max(1, round(kernelsize/8));     % e.g., 8
attn_dim = 128;
numHeads = 8;

r1 = 1;  % dec1_ups
r2 = 1;  % dec2_ups
r3 = 1;  % dec3_ups

lg = layerGraph();

%% ================== Q: Seismic (sequence) ==================
Q1 = [
    sequenceInputLayer(1,'Normalization','zscore','Name','seismic_in')
    ];
lg = addLayers(lg, Q1);

Q2 = [
    convolution1dLayer(k, numFilters, Padding="same", Name="s_conv1")
    batchNormalizationLayer(Name="s_bn1")
    swishLayer(Name="s_sw1")

    residualBlockLayer(numFilters,1,k,true,"s_res1")
    residualBlockLayer(numFilters*2,1,k2,true,"s_res2")
    residualBlockLayer(numFilters*4,1,k10,true,"s_res3")

    convolution1dLayer(1, attn_dim, Padding="same", Name="q_proj")
    batchNormalizationLayer(Name="q_bn")
    swishLayer(Name="q_act")                           % X
    
    % functionLayer(@(X) X, 'Formattable', true, 'Name', 'q_posenc')
    ];

lg = addLayers(lg, Q2);

PE = [
    sinusoidalPositionEncodingLayer(attn_dim, "Name","q_posenc")  % PE
    ];

lg = addLayers(lg,PE);

% X + PE
lg = addLayers(lg, additionLayer(2, "Name","q_pe_add"));

%% ================== K/V: Structure (feature) =================
KV = [
    featureInputLayer(4,'Normalization','zscore','Name','struct_in')
    
    fullyConnectedLayer(32,'Name','kv_fc1282')
    batchNormalizationLayer('Name','kv_bn2')
    swishLayer('Name','kv_feat2')

    fullyConnectedLayer(attn_dim,'Name','kv_fc128')
    batchNormalizationLayer('Name','kv_bn')
    swishLayer('Name','kv_feat1')
    
    functionLayer(@(X) cbtocbt(X),Formattable=true, Acceleratable=true, Name = "kv_feat")
];
lg = addLayers(lg, KV);
lg = addLayers(lg, fullyConnectedLayer(attn_dim, 'Name',"k_fc"));
lg = addLayers(lg, fullyConnectedLayer(attn_dim, 'Name',"v_fc"));

lg = addLayers(lg, functionLayer(@cb2cbt1, 'Name','cb2cbt1_K', 'Formattable',true,'Acceleratable',false));
lg = addLayers(lg, functionLayer(@cb2cbt1, 'Name','cb2cbt1_V', 'Formattable',true,'Acceleratable',false));

%% ================== Attention + Post =====================
Attn = [
    attentionLayer(numHeads,"Name","attention")
    fullyConnectedLayer(attn_dim, "Name","attn_fcout")
];
lg = addLayers(lg, Attn);
lg = addLayers(lg, additionLayer(2,'Name','attn_add'));   % residual add

Post = [
    layerNormalizationLayer('Name','post_ln')
    fullyConnectedLayer(4*attn_dim, 'Name','ffn_fc1')
    swishLayer('Name','ffn_swish')
    % dropoutLayer(0.1, 'Name','ffn_do')
    fullyConnectedLayer(attn_dim, 'Name','ffn_fc2')
];
lg = addLayers(lg, Post);
lg = addLayers(lg, additionLayer(2,'Name','ffn_add'));    % residual add

%% =================== Decoder (flat, no conditionals) =====================
Dec = [
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
lg = addLayers(lg, Dec);

%% ======================= Connections (최소) =========================
lg = connectLayers(lg,'seismic_in','s_conv1');
lg = connectLayers(lg,'seismic_in','q_posenc');
% Q: X + PE
lg = connectLayers(lg, "q_act",    "q_pe_add/in1");
lg = connectLayers(lg, "q_posenc", "q_pe_add/in2");

% K/V → attention (key, value)
lg = connectLayers(lg, "kv_feat",   "k_fc");
lg = connectLayers(lg, "kv_feat",   "v_fc");
lg = connectLayers(lg, "k_fc",      "cb2cbt1_K");
lg = connectLayers(lg, "v_fc",      "cb2cbt1_V");
lg = connectLayers(lg, "cb2cbt1_K", "attention/key");
lg = connectLayers(lg, "cb2cbt1_V", "attention/value");

% Q (X+PE) → attention/query
lg = connectLayers(lg, "q_pe_add",  "attention/query");

% Residual 1 (attn block)
lg = connectLayers(lg, "attn_fcout","attn_add/in1");
lg = connectLayers(lg, "q_pe_add",  "attn_add/in2");

% Post-Attn FFN + Residual 2
lg = connectLayers(lg, "attn_add", "post_ln");
lg = connectLayers(lg, "ffn_fc2",  "ffn_add/in1");
lg = connectLayers(lg, "attn_add", "ffn_add/in2");

% Decoder 
lg = connectLayers(lg, "ffn_add", "dec1_ups");

% Build
net = dlnetwork(lg);
end

%% ================= Residual block (for Q) =================
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

%% ============ Reshape: CB → CBT(T=1) for attention K/V ============
function Y = cb2cbt1(X)
% Input:  X (CB)  [C × B]
% Output: Y (CBT) [C × B × 1]  -- Global conditioning over time
    if isa(X,'dlarray'), Xn = stripdims(X); else, Xn = X; end
    C = size(Xn,1); B = size(Xn,2);
    Y = dlarray(reshape(Xn, C, B, 1), 'CBT');
end

%% ============ Nearest-neighbor upsample by integer r ============
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