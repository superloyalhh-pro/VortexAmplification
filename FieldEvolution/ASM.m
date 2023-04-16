function Output=ASM(Input,Lambda,distance1,F1)

FE = fft2(Input);                                %对场强进行二维傅里叶变换
FE = fftshift(FE);                               %进行频移
phasechange=exp(-1j*2*pi/Lambda*distance1*sqrt(1-F1*Lambda^2));  %计算相移
FFE=FE.*phasechange;                             %计算传输后的场强表达式
E_pre=ifftshift(FFE);                            %进行频移
Output=ifft2(E_pre);                             %二维傅里叶逆变换求出传输后的场强分布

end