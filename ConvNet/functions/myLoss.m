classdef myLoss < dagnn.Loss
 
  properties (Transient)
    
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      d_errors = gather(inputs{1})-(inputs{2});
      fs = size(inputs{1},3);
      bs = size(inputs{1},4);
      outputs{1} = 1/2 * sum((sum(d_errors.^2)));
      n = obj.numAveraged ;
      m = n + fs * bs ;
      obj.average = (n * obj.average + outputs{1}) / m ;
      obj.numAveraged = m ;      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      d_errors = gather(inputs{1})-(inputs{2});
      fs = size(inputs{1},3);
      dx = d_errors/fs;
      dx = gpuArray(dx);
      derInputs{1} = dx;
      derInputs{2} = [] ;
      derParams = {} ;
    end
    
    function obj = myLoss(varargin)
      obj.loss = 'mse';
      obj.load(varargin) ;
    end
  end
end
