classdef myAccuracy < dagnn.Loss
  
  % correlation coefficient
  properties (Transient)
    M = 0
    corCoef = 0
    ssimImg = 0
    E0 = 0
  end
    
  methods
      
      function init(obj)
        lambda = 1030e-9;           % 波长为1030nm
        k = 2*pi/lambda;            % 波数
        w0 = 500e-6;                % 光斑尺寸
        z = 0;                      % 传播距离
        
        N = 128;                    % 设置分辨率
        x = linspace(-3*w0,3*w0,N);     y = x;
        [X,Y] = meshgrid(x,y);
        [theta,r] = cart2pol(X,Y);
        Z_R = pi*w0^2/lambda;       % 瑞利长度
        w_z = w0*sqrt(1+(z/Z_R)^2); % 光束在z位置的半径 
        
        E=@(p,l)sqrt(2*factorial(p)/(pi*factorial((p+abs(l)))))/w_z*(sqrt(2)*r/w_z).^abs(l)...
            *(-1)^p.*obj.laguerre(p,abs(l),2*r.^2/w_z^2).*exp(-r.^2/w_z^2)...
            .*exp(-1i*k*r.^2*z/2/(z^2+Z_R^2)).*exp(1i*(2*p+abs(l)+1)*atan(z/Z_R)).*exp(-1i*l*theta).*exp(1i*k*z);
        
        obj.E0 = zeros(128, 128, N);
        for ii=1:obj.M
            obj.E0(:,:,ii) = E(0,ii-ceil(obj.M(1)/2)) / max(max(abs(E(0,ii-1) )));
        end
      end

      function result = laguerre(obj,p,l,x)
        result = 0;
        if p == 0
            result = 1;
        elseif p == 1
            result = 1+abs(l)-x;
        else
            result = (1/p)*((2*p+l-1-x).*laguerre(p-1,abs(l),x)-(p+l-1)*laguerre(p-2,abs(l),x));
        end
      end

     function EE = LGBeam(obj, ratio)
        if ~obj.M
            obj.M = size(ratio);
            obj.init();
        end
        EE=0;
        for ii=1:obj.M
            EE = EE+obj.E0(:,:,ii)*ratio(ii);
        end
     end
     
     % inputs are labels, params are output result
     function outputs = forward(obj, inputs, params)
        predicts = gather(inputs{1,1});
        labels = gather(inputs{1,2});
        for ii = 1:size(inputs{2},4)
            EE_label = obj.LGBeam(squeeze(labels(1,1,:,ii)));    
            EE_predict = obj.LGBeam(squeeze(predicts(1,1,:,ii)));
            I_label = (EE_label).*conj(EE_label);  I_label= I_label/max(max(I_label));  %归一化
            I_predict = (EE_predict).*conj(EE_predict);  I_predict= I_predict/max(max(I_predict));  %归一化
            obj.corCoef = obj.corCoef+corr2(I_label, I_predict);
            obj.ssimImg = obj.ssimImg+ssim(I_label, I_predict);
        end
        
        bs = size(inputs{2},4);
        obj.numAveraged = obj.numAveraged + bs;
        outputs{1} = obj.average ;
     end
     
     function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs{1} = [] ;
        derInputs{2} = [] ;
        derParams = {} ;
     end
     
     function reset(obj)
        obj.corCoef = 0;
        obj.ssimImg = 0;
        obj.average = [0;0] ;
        obj.numAveraged = 0 ;
     end
     
     function recalacc(obj)
        obj.corCoef = obj.corCoef/obj.numAveraged;
        obj.ssimImg = obj.ssimImg/obj.numAveraged;
        obj.average = [obj.corCoef; obj.ssimImg] ;
     end
      
     function obj = myAccuracy(varargin)
        obj.load(varargin) ;
     end
   end
    
    
end

