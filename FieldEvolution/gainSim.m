clear
close all
clc
% 定义仿真坐标
% 单位是m
lambda=1030e-9;
k = 2*pi/lambda;
Lx=4e-3;                                %定义x方向的范围，范围大一点可以使傅里叶变换不溢出
Ly=Lx;                                  %定义y方向的范围
Nx=2^8;                                 %设定x方向的分段数
Ny=2^8;                                 %设定y方向的分段数
dx=Lx/Nx;                               %设定x方向的取样间隔
dy=Ly/Ny;                               %设定y方向的取样间隔
x=-Lx/2+dx/2:dx:Lx/2-dx/2;              %设定x的取值[1,Nx]
y=-Ly/2+dy/2:dy:Ly/2-dy/2;              %设定y的取值[1,Ny]
[x1,y1]=meshgrid(x,y);                  %生成网格[Ny,Nx]
dfx=1/Lx;                                  %设置x方向的傅里叶变换间隔
fx=-Nx/2*dfx:dfx:(Nx/2-1)*dfx;             %设置x方向的傅里叶变换范围
dfy=1/Ly;                                  %设置y方向的傅里叶变换间隔
fy=-Ny/2*dfy:dfy:(Ny/2-1)*dfy;             %设置y方向的傅里叶变换范围
[fxx,fyy]=meshgrid(fx,fy);                 %生成网格Nfy*Nfx
F = fxx.^2+fyy.^2;

MirrorRad1=1e-3;
MirrorRad2=1e-3;
GainRad=0.5e-3;
D2_square = x1.^2 + y1.^2;
D2 = sqrt(D2_square);

mask1 = zeros(Nx,Ny,'double');
mask1(D2 < MirrorRad1)=1;
mask2 = zeros(Nx,Ny,'double');
mask2(D2 < MirrorRad2)=1;
mask3 = zeros(Nx,Ny,'double');
mask3(D2 < 2*GainRad)=1;

f1 = 550e-3;%第一个透镜的焦距
f2 = 800e-3;%第二个透镜的焦距
fMr = 450e-3;%反射镜曲率
L1 = 270e-3;
L2 = 9700e-3;
L3 = 360e-3;
lcrystal=3*10^(-3);
ROC = 0.9999;

[theta,r] = cart2pol(x1,y1);    m = 2;      w0 = 0.4e-3;    K=2;
% Start = 1.*exp(1j*pi*(rand(Nx,Ny)-1/2)*K);  Ii = real(Start); %初始场为随机相位
% Start = 1.*exp(-(x1.^2+y1.^2)/w0^2); Ii = Start.*conj(Start); %初始场为高斯场分布
Start = (r/w0).^abs(m).*exp(-r.^2/w0^2).*exp(-1i*m*theta); Ii = Start.*conj(Start);  %初始场为涡旋光
figure(1);surf(x1*1e3,y1*1e3,Ii);shading interp; view([0,90]); colormap('hot');

% gi = exp(-(x1.^2+y1.^2)/(GainRad^2));
m = 2;    gi = (r/GainRad).^abs(m).*exp(-r.^2/GainRad^2).*exp(-1i*m*theta); gi=gi.*conj(gi).*mask3;
figure(2);surf(x1*1e3,y1*1e3,gi);shading interp; view([0,90]); colormap('hot');

Circr = Start;
Iteration = 50; %迭代次数.
 for roundtrip = 1:Iteration
     %向右传
     Circr = ASM(Circr,lambda,L1,F); %从M1传到晶体   
     Circr = Circr.*gi.*mask3;
     Circr = Circr.*exp(-1j*k*((x1).^2+(y1).^2)/(2*f1)); %transmite a lens f1
     Circr = ASM(Circr,lambda,L2,F); %从f1传到Len
     Circr = Circr.*exp(-1j*k*((x1).^2+(y1).^2)/(2*f2)); %transmite a lens f2
     Circr = ASM(Circr,lambda,L3,F);

     Circl = Circr.*ROC;%.*mask1;
     Circl = Circl/max(max(Circl));
     
     %向左传
     Circl = ASM(Circl,lambda,L3,F);
     Circl = Circl.*exp(-1j*k*((x1).^2+(y1).^2)/(2*f2)); %transmite a lens f2
     Circl = ASM(Circl,lambda,L2,F); %从f2传到Len
     Circl = Circl.*exp(-1j*k*((x1).^2+(y1).^2)/(2*f1)); %transmite a lens f1
     Circl = Circl.*gi;%.*mask3;%向左增益
     Circl = ASM(Circl,lambda,L1,F); %传到M1
     Circl = Circl.*exp(-1j*k*((x1).^2+(y1).^2)/(2*fMr));%传到Mirror
     Circr = Circl.*ROC.*mask2;
     Circr = Circr/max(max(Circr));

     if (rem(roundtrip,Iteration/100) == 0)
        fprintf('\b\b\b\b\b\b\b\b  %-3.0i %% ',100*roundtrip/Iteration)
     end

 end

Ie = Circr.*conj(Circr);       
II = Ie((2^8-128)/2+1:(2^8+128)/2,(2^8-128)/2+1:(2^8+128)/2);
II = II / max(max(II));
figure(3);
imagesc(II);    colormap('hot');  axis square;  axis off;
% saveas(gcf,'种子光涡旋-泵浦光高斯.png');
save('Detect.mat','II');
% surf(x1*1e3,y1*1e3,Ie);shading interp; view([0,90]); colormap('hot'); 
% axis off;
% title('w0=500um');

