% h=fspecial('gaussian',9,1);%������˹��ͨ�˲���
% I1=filter2(h,im2double(U_net_test_pre));    % ��ͼ��ת����double�������ͣ�0~1��

%��ȡU_netԤ��ͼ
U_net_test_pre=imread('./assess/Unet/test/4104.bmp');
I1=im2double(U_net_test_pre); 
% imshow(I1);
% figure(1);
% xlabel('x');
% ylabel('y');
% colorbar  %��ʾ��ɫ��
% axis on;
% axis normal;
% figure(2);
% surf(I1);
% colorbar  %��ʾ��ɫ��
% shading interp




% % ��ȡNestedԤ��ͼ
Nested_test_pre=imread('./assess/Nestednet/test/4104.bmp');
I2=im2double(Nested_test_pre);    % ��ͼ��ת����double�������ͣ�0~1��
% figure(11);
% imshow(I2);
% colorbar  %��ʾ��ɫ��
% figure(12);
% surf(I2);
% shading interp


% %��ȡunet3plusԤ��ͼ
Unet3plus_test_pre=imread('./assess/Unet3plus/test/4104.bmp');
I3=im2double(Unet3plus_test_pre);    % ��ͼ��ת����double�������ͣ�0~1��
% figure(31);
% imshow(I3);
% colorbar  %��ʾ��ɫ��
% figure(32);
% surf(I3);


%��ȡ��ʵͼ
test_real=imread('./assess/real/4104.bmp');
I5=im2double(test_real);    % ��ͼ��ת����double�������ͣ�0~1��
% figure(51);
% surf(I5);
% shading interp



%SSIM1:Unet����ʵͼssim %SSIM2:Nested����ʵͼssim %SSIM3:Res_net����ʵͼssim %SSIM4:Unet3plus����ʵͼssim
 
SSIM1=ssim(U_net_test_pre,test_real);
fprintf('\nSSIM1:%f\n',SSIM1);
SSIM2=ssim(Nested_test_pre,test_real);
fprintf('\nSSIM2:%f\n',SSIM2);
SSIM3=ssim(Unet3plus_test_pre,test_real);
fprintf('\nSSIM3:%f\n',SSIM3);


%A��Unet����ʵͼ���  AA��Nested����ʵͼ��� AAA:Unet3plus����ʵͼ���  
% A=I1-I5;
% figure(111);
% surf(A);
% AA=I2-I5;
% figure(222);
% surf(AA);
% AAA=I3-I5;
% figure(333);
% surf(AAA);
