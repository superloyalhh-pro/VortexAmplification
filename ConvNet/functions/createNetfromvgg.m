function net = createNetfromvgg()

%This is the orginal VGG-16 model, which can be acquired from http://www.vlfeat.org/matconvnet/
vggnet = load('model\imagenet-vgg-verydeep-16.mat') ; 
net.layers = vggnet.layers(1:33);

net.layers{1}.size = [3, 3, 1, 64];
net.layers{1}.weights{1}= 0.01*randn(3,3,1,64, 'single');
net.layers{1}.weights{2} = zeros(1,64,'single');

net.layers{end-1}.weights{1} = 0.01*randn(4,4,512,1024, 'single');
net.layers{end-1}.weights{2} = zeros(1,1024,'single');

end

