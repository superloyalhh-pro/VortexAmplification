clc
clear 
close all
%% 拉盖尔-高斯光束,单位m
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
    *(-1)^p.*laguerre(p,abs(l),2*r.^2/w_z^2).*exp(-r.^2/w_z^2)...
    .*exp(-1i*k*r.^2*z/2/(z^2+Z_R^2)).*exp(1i*(2*p+abs(l)+1)*atan(z/Z_R)).*exp(-1i*l*theta).*exp(1i*k*z);

%% 生成L个模式混合M张图片
L = 5;
M = 1000;

label=zeros(L,M);
pattern=zeros(N,N,M,'uint8');

for ii = 1:M
    if mod(ii,100)==0   
        fprintf('Generated %d imgs...\n', ii);  
    end
    
    E0 = 0;
    xx = rand(1, L);
    yy = randNum(L);
    for jj = 1:L
        E1 = E(0,jj-1);    E1=E1/max(max(abs(E1)));
        E0 = E0+yy(jj)*E1;
    end
    I0 = (E0).*conj(E0);  I0= I0/max(max(I0))*255;  %归一化到255
    label(:,ii)=yy';
    pattern(:,:,ii)=I0;
end

% 保存数据和标签
% train_label = label;
% train_pattern = pattern;
% save('train_data_5mode.mat','train_label','train_pattern');

test_label = label;
test_pattern = pattern;
save('test_data_5mode.mat','test_label','test_pattern');

figure(1);
imagesc(pattern(:,:,1));
title('Vortex beam','fontname','times new Roman','fontsize',16);
axis square;    axis off;
colormap hot; colorbar;

%% 拉盖尔多项式
function result = laguerre(p,l,x)
    result = 0;
    if p == 0
        result = 1;
    elseif p == 1
        result = 1+abs(l)-x;
    else
        result = (1/p)*((2*p+l-1-x).*laguerre(p-1,abs(l),x)-(p+l-1)*laguerre(p-2,abs(l),x));
    end
end

%% 生成和为1的L个随机数
function yy = randNum(L)
    r = sort(rand(1, L-1)); % 生成L-1个随机数，并排序
    intervals = [r 1] - [0 r]; % 计算每个数的间隔
    yy = intervals / sum(intervals); % 得到L个和为1的数
end

