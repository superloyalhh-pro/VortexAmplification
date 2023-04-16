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

%% 两个光束的这个径向阶数p和拓扑荷数l
p1 = 0; l1 = -1; 
E1 = E(p1,l1);      E1 = E1/max(max(abs(E1)));
I1 = (E1).*conj(E1);  I1= I1/max(max(I1));  %归一化

p2 = 0; l2 = 1; 
E2 = E(p2,l2);      E2 = E2/max(max(abs(E2)));
I2 = (E2).*conj(E2);  I2= I2/max(max(I2));  %归一化

alpha = 0.5;
fai = -2*pi*0;
E3 = alpha*E1 + (1-alpha)*exp(1i*fai)*E2;
% E3 = sqrt(alpha)*(E1/max(max(abs(E1)))) + sqrt(1-alpha)*(exp(1i*fai)*E2/max(max(abs(E2))));
I3 = (E3).*conj(E3);  I3= I3/max(max(I3));  %归一化

%% 绘图
scrsz = get(0,'screensize');
figure('position',[scrsz(3)*1/4,scrsz(4)*1/4,scrsz(3)/2,scrsz(4)/2])
subplot(2,2,1);
imagesc(I1);
title('Vortex beam','fontname','times new Roman','fontsize',16);
axis square;    axis off;
colormap hot; colorbar;

subplot(2,2,2);
imagesc(I2);
title('Gaussian beam','fontname','times new Roman','fontsize',16);
axis square;    axis off;
colormap hot; colorbar;

subplot(2,2,3);
imagesc(I3);
title('Mixed beam','fontname','times new Roman','fontsize',16);
axis square;    axis off;
colormap hot; colorbar;

subplot(2,2,4);
phase = angle(E1)./max(max(angle(E1)));
imagesc(phase)
title('Phase','fontname','times new Roman','fontsize',16);
axis square;    axis off;
colormap hot; colorbar;
print('subplot_image.png', '-dpng')

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
