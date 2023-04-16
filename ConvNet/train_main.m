function train_main()
addpath(genpath('functions'));
mySetup();

% some example settings
trainOpts.id = 5;
trainOpts.useGpu = true;  
trainOpts.epoch_start = 0;  % continue training with epoch_start
trainOpts.numEpochs = 15;
trainOpts.batchSize = 64;
trainOpts.dataDir = 'model';
trainOpts.saveDir = 'save' ;
trainOpts.outputDir = 'output' ;

trainOpts.learningRate = ones(1,50) ;
trainOpts.learningRate(1,1:20)= 1;
trainOpts.learningRate(1,21:50)= 0.1;

my_cnn_train_dag(getBatchWrapper, trainOpts) ;

function fn = getBatchWrapper()
fn = @(imdb,batch,opts) getBatch(imdb,batch,opts) ;
