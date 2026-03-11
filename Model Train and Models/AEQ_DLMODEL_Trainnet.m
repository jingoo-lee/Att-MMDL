clc; clear all; close all;
%% 1 dataset loading
load('206multimodaldata_single.mat')
rng(123,'twister')

gm1 = reshape(tot_gm,3000,1,[]); % 3000 * 1* 2060
rv = rv(:,:)'; % 4 * 2060
response1 = reshape(data,3000,139,[]); % 3000 * 139 * 2060

numtrain = floor(0.8*(size(rv,2)));
numtrain = 1600;
%% 2    
% index = randperm(139,60);
index = [3	4	6	7	13	14	20	23	24	27	29	33	35	38	40	43	44	45	47	52	53	56	57	58	61	62	64	65	71	73	76	78	82	84	85	88	91	95	96	98	99	101	105	107	108	109	111	116	118	119	120	121	122	123	126	128	129	130	133	134];
xtrain1 = arrayDatastore(permute(gm1(:,:,1:numtrain),[1 2 3]),"IterationDimension",3);
xtrain2 = arrayDatastore(rv(:,1:numtrain),"IterationDimension",2);
ytrain = arrayDatastore(permute(response1(:,index,1:numtrain),[1 2 3]),'IterationDimension',3);
dsTrain = combine(xtrain1,xtrain2,ytrain);

%%
kernelsize = 64;
[layers] = Baselinemodel_LSTM(kernelsize);
[layers] = Baselinemodel_ResCNN(kernelsize);
[layers] = Proposedmodel_RESS(kernelsize);
analyzeNetwork(layers)
%% 3
options = trainingOptions('adam', ...
    'MaxEpochs',1400, ...
    'InitialLearnRate',1e-3, ...
    'GradientThresholdMethod','l2norm',...
    'InputDataFormats',{'TCB','CB'},...
    'L2Regularization',1e-3,...
    'MiniBatchSize',256,...
    'GradientThreshold',1,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',300,... 
    'VerboseFrequency',30,...
    'LearnRateDropFactor',0.9,...
    'ExecutionEnvironment','auto',...
    'Acceleration','auto',...
    'Verbose',true, ...
    'Plots','none');
%% 4
net = trainnet(dsTrain,layers,'mse',options);
%% 6
xtest1 = arrayDatastore(permute(gm1(:,:,numtrain+1:end),[1 2 3]),"IterationDimension",3);
xtest2 = arrayDatastore(rv(:,numtrain+1:end),"IterationDimension",2);
dsTest = combine(xtest1,xtest2);
%% 7
prediction = minibatchpredict(net,dsTest,"InputDataFormats",{'TCB','CB'}); %% 300 * 30730

%% 7
% pred = prediction'; %% 300*3*6*2195
refer = squeeze(response1(:,index,numtrain+1:end));
pred = prediction;
pred = reshape(pred,3000,size(index,2),[]);

%% 4 mMAPE
for j=1:60
    mMAPE = zeros(460, 1); % mMAPE 초기화
    for i=1:460
        Pred = pred(:,j,i);
        Test = refer(:,j,i);
        A_max = max(abs(Test));
        if A_max < 1
            mMAPE(i) = 100*mean(abs(Test-Pred));
        else
            mMAPE(i) = 100*mean(abs(Test-Pred)/A_max);
        end
    end
    mape_matrix(j) = mean(mMAPE);
    tot_mape_matrix(j,:) = mMAPE;
end

avg_mape = mean(mape_matrix)

bar(mape_matrix)
xlabel('Node number')
ylabel('mMAPE (%)')
legend(['Mean mMAPE = ', num2str(avg_mape, '%.2f'), '%'])
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman')


%% test plot

%%
reference = double(reshape(refer, [], 1));
prediction = double(reshape(pred, [], 1));


SS_res = sum((reference - prediction).^2);
SS_tot = sum((reference - mean(reference)).^2);
R2 = 1 - SS_res / SS_tot;


nbins = 500;
edges_x = linspace(min(reference), max(reference), nbins);
edges_y = linspace(min(prediction), max(prediction), nbins);
[counts, edgesX, edgesY] = histcounts2(reference, prediction, edges_x, edges_y);
counts(counts == 0) = NaN;  % 0인 부분은 표시하지 않음


X = 0.5 * (edgesX(1:end-1) + edgesX(2:end));
Y = 0.5 * (edgesY(1:end-1) + edgesY(2:end));


xrange = prctile(reference, [0.01 99.99]);
yrange = prctile(prediction, [0.01 99.99]);

f3 = figure;
f3.Position(1:2) = [100 100];
f3.Color = [1 1 1];

contourf(X, Y, log10(counts'), 20, 'LineColor', 'none');
axis xy;
colormap('parula');
colorbar;
% caxis([1.5, max(log10(counts(:)), [], 'omitnan')]);

hold on;

line(xrange, xrange, 'Color', [0 0 0 0.3], 'LineWidth', 0.3);


x_text = xrange(1) + 0.05 * range(xrange);
y_text = yrange(2) - 0.1 * range(yrange);

text(x_text, y_text, sprintf('R^2 = %.4f', R2), ...
    'FontSize', 14, 'FontName', 'Times New Roman', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

set(gca, 'FontSize', 13, 'FontName', 'Times New Roman', 'LineWidth', 1.2);
axis equal;
xlim(xrange);
ylim(yrange);  % equal 때문에 다시 한 번 보정
box on;
grid off;
hold off;
%%
currentDate = datestr(now, 'yymmdd'); 

folderName = sprintf('%s_Proposed_mMAPE_%.5f', currentDate, avg_mape);

if ~exist(folderName, 'dir')
    mkdir(folderName); 
end

filename_net = fullfile(folderName, sprintf('net.mat'))
save(filename_net, 'net','refer','pred','index');
close all;