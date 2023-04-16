function net = myDAGnet

net = createNetfromvgg();
net = vl_simplenn_tidy(net) ;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% net.layers(2).block = dagnn.Sigmoid() ;
% net.layers(4).block = dagnn.Sigmoid() ;
% net.layers(7).block = dagnn.Sigmoid() ;
% net.layers(9).block = dagnn.Sigmoid() ;
% net.layers(12).block = dagnn.Sigmoid() ;
% net.layers(14).block = dagnn.Sigmoid() ;
% net.layers(16).block = dagnn.Sigmoid() ;
% net.layers(19).block = dagnn.Sigmoid() ;
% net.layers(21).block = dagnn.Sigmoid() ;
% net.layers(23).block = dagnn.Sigmoid() ;
% net.layers(26).block = dagnn.Sigmoid() ;
% net.layers(28).block = dagnn.Sigmoid() ;
% net.layers(30).block = dagnn.Sigmoid() ;
% net.layers(33).block = dagnn.Sigmoid() ;

% net.layers(5).block.method = 'avg';
% net.layers(10).block.method = 'avg';
% net.layers(17).block.method = 'avg';
% net.layers(24).block.method = 'avg';
% net.layers(31).block.method = 'avg';

net.addLayer('fc7', ...
     dagnn.Conv('size', [1 1 1024 5], 'pad', [0 0 0 0]), ...
     'x33', 'x34', {'fc7_f','fc7_b'});

f = net.getParamIndex('fc7_f') ;
net.params(f).value = 0.01*randn(1, 1, 1024, 5, 'single') ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('fc7_b') ;
net.params(f).value = zeros(1, 1, 5, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

softmaxBlock = dagnn.SoftMax() ;
net.addLayer('softmax_m1', softmaxBlock, 'x34', 'prediction', {}) ;

% Add loss layer
net.addLayer('objective', ...
  myLoss(), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  myAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;


end

