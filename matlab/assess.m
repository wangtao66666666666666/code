% h=fspecial('gaussian',9,1);%产生高斯低通滤波器
% I1=filter2(h,im2double(U_net_test_pre));    % 把图像转换成double精度类型（0~1）

%读取U_net预测图
U_net_test_pre=imread('./assess/Unet/test/4104.bmp');
I1=im2double(U_net_test_pre); 
% imshow(I1);
% figure(1);
% xlabel('x');
% ylabel('y');
% colorbar  %显示颜色栏
% axis on;
% axis normal;
% figure(2);
% surf(I1);
% colorbar  %显示颜色栏
% shading interp




% % 读取Nested预测图
Nested_test_pre=imread('./assess/Nestednet/test/4104.bmp');
I2=im2double(Nested_test_pre);    % 把图像转换成double精度类型（0~1）
% figure(11);
% imshow(I2);
% colorbar  %显示颜色栏
% figure(12);
% surf(I2);
% shading interp


% %读取unet3plus预测图
Unet3plus_test_pre=imread('./assess/Unet3plus/test/4104.bmp');
I3=im2double(Unet3plus_test_pre);    % 把图像转换成double精度类型（0~1）
% figure(31);
% imshow(I3);
% colorbar  %显示颜色栏
% figure(32);
% surf(I3);


%读取真实图
test_real=imread('./assess/real/4104.bmp');
I5=im2double(test_real);    % 把图像转换成double精度类型（0~1）
% figure(51);
% surf(I5);
% shading interp



%SSIM1:Unet与真实图ssim %SSIM2:Nested与真实图ssim %SSIM3:Res_net与真实图ssim %SSIM4:Unet3plus与真实图ssim
 
SSIM1=ssim(U_net_test_pre,test_real);
fprintf('\nSSIM1:%f\n',SSIM1);
SSIM2=ssim(Nested_test_pre,test_real);
fprintf('\nSSIM2:%f\n',SSIM2);
SSIM3=ssim(Unet3plus_test_pre,test_real);
fprintf('\nSSIM3:%f\n',SSIM3);


%A；Unet与真实图误差  AA；Nested与真实图误差 AAA:Unet3plus与真实图误差  
% A=I1-I5;
% figure(111);
% surf(A);
% AA=I2-I5;
% figure(222);
% surf(AA);
% AAA=I3-I5;
% figure(333);
% surf(AAA);
