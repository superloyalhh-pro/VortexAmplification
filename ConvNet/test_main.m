close all
clear all
modeNum=[5];
%% Fig4 Averaged prediction error as a function of epochs for 4 cases
Loss = zeros(4,2,15);
corCoef = zeros(4,2,15);
psnrImg = zeros(4,2,15);
for ii = 1:length(modeNum)
    table = xlsread(sprintf('output\\VGG-info-%d.xlsx', modeNum(ii)));
    Loss(ii,1,:)=table(:,1);    Loss(ii,2,:)=table(:,4);
    corCoef(ii,1,:)=table(:,2); corCoef(ii,2,:)=table(:,5);
    psnrImg(ii,1,:)=table(:,3); psnrImg(ii,2,:)=table(:,6);
end
x=[1:1:15]; %x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
curveSet={'-s', '-^', '-v', 'o-'};
colorSet = [142, 207, 201; 255, 190, 122; 250, 127, 111; 130, 176, 210]/255;

% plot Training Loss Curve
figure();
for ii = 1:length(modeNum)
    plot(x,squeeze(Loss(ii,1,:)),curveSet{ii}, 'Linewidth', 1.3, 'markersize', 3, 'Color', colorSet(ii,:)); hold on; %线型，标记，颜色  
    str{ii} = [sprintf('%d-mode-mixed',modeNum(ii))];
end
grid on;
axis([0,20,0,0.0075])  %确定x轴与y轴框图大小
set(gca,'XTick',[0:4:20]) %x轴范围，间隔minZhi
set(gca,'YTick',[0:0.001:0.007]) %y轴范围，间隔minZhi
set(gca, 'FontSize', 12);
set(gca, 'FontName', 'Times New Roman');
legend(str, 'location', 'NorthEast', 'FontSize', 12, 'FontName', 'Times New Roman', 'box', 'off');   %右下角标注
xlabel('Epochs','fontsize',14,'FontName','Times New Roman')   %x轴坐标描述
ylabel('Training Loss', 'fontsize',14,'FontName','Times New Roman') %y轴坐标描述
saveas(gcf,'Fig\TrainingLossCurve.png');

% plot Trainging Correlation Curve
figure();
for ii = 1:length(modeNum)
    plot(x,squeeze(corCoef(ii,1,:)),curveSet{ii}, 'Linewidth', 1.3, 'markersize', 3, 'Color', colorSet(ii,:)); hold on; %线型，标记，颜色  
    str{ii} = [sprintf('%d-mode-mixed',modeNum(ii))];
end
grid on;
axis([0,20,0.85,1])  %确定x轴与y轴框图大小
set(gca,'XTick',[0:4:20]) %x轴范围，间隔minZhi
set(gca,'YTick',[0.85:0.03:1]) %y轴范围，间隔minZhi
set(gca, 'FontSize', 12);
set(gca, 'FontName', 'Times New Roman');
legend(str, 'location', 'SouthEast', 'FontSize', 12, 'FontName', 'Times New Roman', 'box', 'off');   %右下角标注
xlabel('Epochs','fontsize',14,'FontName','Times New Roman')   %x轴坐标描述
ylabel('Training Correlation', 'fontsize',14,'FontName','Times New Roman') %y轴坐标描述
saveas(gcf,'Fig\TrainingCorrelationCurve.png');

% plot Testing Correlation Curve
figure();
for ii = 1:length(modeNum)
    plot(x,squeeze(corCoef(ii,2,:)),curveSet{ii}, 'Linewidth', 1.3, 'markersize', 3, 'Color', colorSet(ii,:)); hold on; %线型，标记，颜色  
    str{ii} = [sprintf('%d-mode-mixed',modeNum(ii))];
end
grid on;
axis([0,20,0.85,1])  %确定x轴与y轴框图大小
set(gca,'XTick',[0:4:20]) %x轴范围，间隔minZhi
set(gca,'YTick',[0.85:0.03:1]) %y轴范围，间隔minZhi
set(gca, 'FontSize', 12);
set(gca, 'FontName', 'Times New Roman');
legend(str, 'location', 'SouthEast', 'FontSize', 12, 'FontName', 'Times New Roman', 'box', 'off');   %右下角标注
xlabel('Epochs','fontsize',14,'FontName','Times New Roman')   %x轴坐标描述
ylabel('Testing Correlation', 'fontsize',14,'FontName','Times New Roman') %y轴坐标描述
saveas(gcf,'Fig\TestingCorrelationCurve.png');

%% Load the model
figCom=figure(17);
t = tiledlayout(2,4);
t.Padding = 'compact';
t.TileSpacing = 'compact';

figCom=figure(27);
t2 = tiledlayout(1,4);
t2.Padding = 'compact';
t2.TileSpacing = 'compact';

addpath(genpath('functions'));
mySetup();
for ii = 1:length(modeNum)
    load (sprintf('save/VGG-net-%dmode.mat', modeNum(ii)),'net');
    net = dagnn.DagNN.loadobj(net) ;
    net.move('gpu');
    net.mode = 'test';
    load (sprintf('test_data_%dmode.mat', modeNum(ii))) 
    images(:,:,1,:)=test_pattern;
        
    %% plot some sample images
    figSam=figure(27);
    scrsz = get(0,'screensize');
    figCom.Position = [scrsz(3)*1/5,scrsz(4)*1/4,scrsz(3)*3/5,scrsz(4)*1/4];
    nexttile(ii);   imagesc(test_pattern(:,:,9));
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])
    if ii==1
        shading interp; view([0,90]); colormap hot;
        cb = colorbar;
        cb.Layout.Tile = 'east';
    end
    
    xlabel(sprintf('%d-mode-mixed', modeNum(ii)),...
        'FontWeight','bold','FontSize',12, 'FontName','Times New Roman');
    saveas(gcf,sprintf('Fig\\Sample.png'));


    N = modeNum(ii);
    n=size(test_pattern,3);
    Error = zeros(N,1000);
    for jj = 1:1:n
        im = single(images(:,:,1,jj)) ;    
        im = gpuArray(im);
        % run the CNN
        predVar = net.getVarIndex('prediction');
        inputVar = 'input' ;
        net.eval({inputVar, im});
        tmp = squeeze(double(gather(net.vars(predVar).value))) ;
        output(1:N,jj) = double(tmp(1:N,:));
        Error(:,jj) = abs(output(:,jj)-test_label(:,jj));
    end
    
    %% Fig6 The comparison of label and predicted value for ten typical pattern samples
    figCom=figure(17);
    scrsz = get(0,'screensize');
    figCom.Position = [scrsz(3)*1/5,scrsz(4)*1/4,scrsz(3)*3/5,scrsz(4)*1/2];
    nexttile(ii);   image(test_pattern(:,:,1));
    if ii==1
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        ylabel('Label', 'fontsize',20,'FontName','Times New Roman') %y轴坐标描述
    else
        axis off;
    end
    II=beam(output(:,1), N); 

    nexttile(ii+4);     image(II); 
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])
    if ii==1
        ylabel('Predict', 'fontsize',20,'FontName','Times New Roman') %y轴坐标描述
        shading interp; view([0,90]); colormap hot;
        cb = colorbar;
        cb.Layout.Tile = 'east';
    end
    
    xlabel(sprintf('%d-mode-mixed\nCorr: %.4f, SSIM: %.4f', modeNum(ii), corr2(test_pattern(:,:,1),II), ssim(double(test_pattern(:,:,1)),II)),...
        'FontWeight','bold','FontSize',12, 'FontName','Times New Roman');
    saveas(gcf,sprintf('Fig\\Comparision.png'));

    %% Fig7 Comparison of the label and predicted value for 4 involved cases
    [errData, errLabel] = sort(Error(1,:), 2);
    gaussianRatio(1,:) = output(1, errLabel);
    gaussianRatio(2,:) = test_label(1, errLabel);
    M = 500;    % M points to plot
    [sData, sLabel] = sort(gaussianRatio(2,1:M), 2);
    x=[1:1:M];
    figure();
    plot(1:M,gaussianRatio(2,sLabel(1:M)),'-.k', 1:M,gaussianRatio(1,sLabel(1:M)),'-.r', 'Linewidth', 1.3); %线型，标记，颜色
    grid on;
    axis([0,M,0,1]);  %确定x轴与y轴框图大小
    set(gca,'XTick',[0:M/5:M]) %x轴范围0-N，间隔minZhi
    set(gca,'YTick',[0:0.2:1]) %y轴范围0-3，间隔minZhi
    set(gca, 'FontSize', 12);
    set(gca, 'FontName', 'Times New Roman');
    legend({'Label', 'Predicted'}, 'location', 'NorthEast', 'FontSize', 13, 'FontName', 'Times New Roman', 'box', 'off');   %右下角标注
    xlabel(sprintf('Number (%d-mode-mixed)', modeNum(ii)),'fontsize',15,'FontWeight','bold','FontName','Times New Roman')   %x轴坐标描述
    ylabel('Gaussian Ratio', 'fontsize',15,'FontWeight','bold','FontName','Times New Roman') %y轴坐标描述
    saveas(gcf,sprintf('Fig\\Comparison-%dmode.png', modeNum(ii)));
end

%% 拉盖尔-高斯光束,单位m
function II = beam(yy, L)
lambda = 1030e-9;           % 波长为1030nm
k = 2*pi/lambda;            % 波数
w0 = 500e-6;                % 光斑尺寸
z = 0;

N = 128;                    % 设置分辨率
x = linspace(-3*w0,3*w0,N);     y = x;
[X,Y] = meshgrid(x,y);
[theta,r] = cart2pol(X,Y);
Z_R = pi*w0^2/lambda;       % 瑞利长度
w_z = w0*sqrt(1+(z/Z_R)^2); % 光束在z位置的半径 
% 径向阶数p和拓扑荷数l
E=@(p,l)(-1)^p*sqrt(2*factorial(p)/(pi*factorial((p+abs(l)))))*(1/w_z)*(sqrt(2)*r/w_z).^abs(l)...
    .*exp(-r.^2/w_z^2).*laguerre(p,abs(l),2*r.^2/w_z^2).*exp(-1i*l*theta).*exp(-1i*k*z)...
   .*exp(-1i*k*r.^2*z/2/(z^2+Z_R^2)).*exp(-1i*(2*p+abs(l)+1)*atan(z/Z_R));
EE = 0;
for jj = 1:L
    E1 = E(0,jj-1);   E1=E1/max(max(abs(E1)));
    EE = EE+yy(jj)*E1;
end
EE = exp(1i*2*pi*rand(1))*EE;
II = (EE).*conj(EE);  II= II/max(max(II))*255;  %归一化到255
end

function result = laguerre(p,l,x)
    if p == 0
        result = 1;
    elseif p == 1
        result = 1+abs(l)-x;
    else
        result = (1/p)*((2*p+l-1-x).*laguerre(p-1,abs(l),x)-(p+l-1)*laguerre(p-2,abs(l),x));
    end
end