%四步相移轮廓术的仿真实验
%仿真的步骤分为以下五步
% 
% --------------第一步，三维图形的模拟-------------------
% %模拟的时候以像素作为单位
tic;
x=1:8;
y=1:8;
temps=zeros(8,8)
for i=1:1:8
    for j=1:1:8
        temps(i,j)=rand()*30
    end
end
[x,y]=meshgrid(x,y);
% surf(x,y,temps);
% shading interp
% t=0:0:0;
% set(gca,'xtick',t); %这两句话可以去掉x轴的刻度和坐标值
% xlabel('x轴的说明') %这句话可以坐标的下面添加一个说明
% set(gca,'ytick',t); %这两句话可以去掉y轴的刻度和坐标值
% ylabel('y轴的说明') %这句话可以坐标的下面添加一个说明 
% set(gca,'ztick',t); %这两句话可以去掉y轴的刻度和坐标值
% zlabel('z轴的说明') %这句话可以坐标的下面添加一个说明 
% 三次样条插值
xi=1:0.05:7.35;
yi=1:0.05:7.35;
[xi,yi]=meshgrid(xi,yi);
zi=interp2(x,y,temps,xi,yi,'cubic');
yi = max(max(zi));
mi=zi/yi;
x = 1:128;
y = 1:128; 
figure(1);mesh(x,y,zi);
% title('实际物体的三维轮廓')
% clear indexs;
figure(2);
imshow(mi);
% k=5;
% save_path=['./imgs/',num2str(k),'.bmp'];
% imwrite(mi,save_path,'bmp');%图片的写入，第一个光栅
% img = imread(save_path); % 读入是unit8型(0~255)数据
% I1  = im2double(img);    % 把图像转换成double精度类型（0~1）
% I2  = double(img)/255;   % uint8转换成double,作用同im2double
% figure(3);
% surf(I2)

% colorbar  %显示颜色栏
% figure(4);surf(mi)
% shading interp
% view(0,90) %俯视图xoy平面，x为横轴
% view(-90,0) %侧视图yoz平面，y为横轴
% view(0,0) %侧视图xoz平面，x为横轴
% figure(5)
% colorbar  %显示颜色栏
% colormap(jet)  %颜色的风格选择
%--------------第二步，正弦光栅的制作-------------------
x=1:128;
f = 1/20;%一个光栅空间频率
X = ones(128,1)*x;
fai =2*pi*f.*X;
I00 =1+cos(fai); 
imwrite(I00,'shatter00.bmp','bmp');%图片的写入，没有发生移位

%系统参数的设置
len =100;%相机的出瞳中心到参考面的距离
d =20; %投影中心到照相机中心的距离
p =1/f;%投影光栅的空间周期，6个像素
detafai = 2*pi*f*d.*zi/len;%物体表面增加的相位
clear X Y Z I00;

%变形光栅的写入
I11 = 1 + cos(fai+detafai);%1+cos()
imwrite(I11,'shatter11.bmp','bmp');%图片的写入，第一个光栅
I12 = 1 + cos(fai+detafai+pi/2);% 1-sin()
imwrite(I12,'shatter12.bmp','bmp');%图片的写入，第二个光栅
I13 = 1 + cos(fai+detafai+pi);% 1-cos()
imwrite(I13,'shatter13.bmp','bmp');%图片的写入，第三个光栅
I14 = 1 + cos(fai+detafai+3*pi/2);%1+sin()
imwrite(I14,'shatter14.bmp','bmp');%图片的写入，第四个光栅

%原始光栅的写入
I01 = 1 + cos(fai);%1+cos()
imwrite(I01,'shatter01.bmp','bmp');%图片的写入，第一个光栅
I02 = 1 + cos(fai+pi/2);% 1-sin()
imwrite(I02,'shatter02.bmp','bmp');%图片的写入，第二个光栅
I03 = 1 + cos(fai+pi);% 1-cos()
imwrite(I03,'shatter03.bmp','bmp');%图片的写入，第三个光栅
I04 = 1 + cos(fai+3*pi/2);%1+sin()
imwrite(I04,'shatter04.bmp','bmp');%图片的写入，第四个光栅
clear I01 I02 I03 I04 I11 I12 I13 I14;

%--------------第三步，图片的读取以及初始化滤波-------------------

h=fspecial('gaussian',9,1);%产生高斯低通滤波器

%参考图片与调制图片的读取
%参考图片读取，为了获取载波相位
I01=imread('shatter01.bmp','bmp'); %I01=double(I01);
I01=double(filter2(h,double(I01)));
I02=imread('shatter02.bmp','bmp'); %I02=double(I02);
I02=double(filter2(h,double(I02)));
I03=imread('shatter03.bmp','bmp'); %I03=double(I03);
I03=double(filter2(h,double(I03)));
I04=imread('shatter04.bmp','bmp'); 
I04=double(filter2(h,double(I04)));
%调制光栅图片的读取，为了获取变形后的相位
I11=imread('shatter11.bmp','bmp'); %I11=double(I11);
I11=double(filter2(h,double(I11)));
I12=imread('shatter12.bmp','bmp'); %I12=double(I12);
I12=double(filter2(h,double(I12)));
I13=imread('shatter13.bmp','bmp'); %I13=double(I13);
I13=double(filter2(h,double(I13)));
I14=imread('shatter14.bmp','bmp'); I14=double(filter2(h,double(I14)));
[width,height]=size(I11);%获得图片的尺寸

%phase0是载波相位，phase1是变形条纹后的相位，phase是物体的相位，还需要进行反卷叠处理
%phase0是载波相位，phase1是变形条纹后的相位，phase是物体的相位，还需要进行反卷叠处理
phase1 = zeros(128,128);
phase0 = phase1;
phase = phase0;%相位矩阵的初始化
% ------------------四步相移法
for i=1:width
    for j=1:height
        s = abs(atan((I02(i,j)-I04(i,j))/(I01(i,j)-I03(i,j))));
        if(I02(i,j)>=I04(i,j))%sin(fai)<0
            if(I01(i,j)<=I03(i,j))%cos(fai)>0
                phase0(i,j)= -s;%第四象限
            else
                phase0(i,j)= s - pi;%第三象限
            end
            
        else%sin(fai)>0
            if(I01(i,j)<=I03(i,j))%cos(fai)>0
                phase0(i,j)= s;%第一象限
            else
                phase0(i,j)= pi - s;%第二象限
            end
        end
    end
end

%调制变形后的相位的提取
for i=1:width
    for j=1:height
        s = abs(atan((I12(i,j)-I14(i,j))/(I11(i,j)-I13(i,j))));
        if(I12(i,j)>=I14(i,j))%sin(fai)<0
            if(I11(i,j)<=I13(i,j))%cos(fai)>0
                phase1(i,j)= -s;%第四象限
            else
                phase1(i,j)= s - pi;%第三象限
            end
            
        else%sin(fai)>0
            if(I11(i,j)<=I13(i,j))%cos(fai)>0
                phase1(i,j)= s;%第一象限
            else
                phase1(i,j)= pi - s;%第二象限
            end
        end
    end
end

clear width height s i j detafai fai x y I01 I02 I03 I04 I11 I12 I13 I14
% -----------------解包裹代码
% %两个部分相位的相减，得到最后的相位
P = phase0- phase1;
% % 四向最小二乘相位解包裹算法
[M,N] = size(P);
dx=zeros(M,N); %预定义X方向的梯度
dy=zeros(M,N); %预定义y方向的梯度
du=zeros(M,N); %预定义左下对角线方向的梯度
dv=zeros(M,N); %预定义右下对角线方向的梯度
m=1:M-1;          %通过循环计算梯度
dx(m,:)=P(m+1,:)-P(m,:);
dx=dx-pi*floor(dx/pi+0.5);  %将截断的梯度连接起来
n=1:N-1;
dy(:,n)=P(:,n+1)-P(:,n);
dy=dy-pi*floor(dy/pi+0.5);
m=2:M-1;
n=2:N-1;
dv(m,n)=P(m+1,n-1)-P(m,n);
n=2:N;
dv(1,n)=P(2,n-1)-P(1,n);
dv=dv-pi*floor(dv/pi+0.5);

m=1:M-1;
n=1:N-1;
du(m,n)=P(m+1,n+1)-P(m,n);
du=du-pi*floor(du/pi+0.5);
p=zeros(M,N);               %求ρ(x,y)
p1=zeros(M,N);
p2=zeros(M,N);
p3=zeros(M,N);
p4=zeros(M,N);
m=2:M-1;
p1(m,:)=dx(m,:)-dx(m-1,:);
n=2:N-1;
p2(:,n)=dy(:,n)-dy(:,n-1);
m=2:M-1;
n=2:N-1;
p3(m,n)=dv(m,n)-dv(m-1,n+1);
m=2:M-1;
n=2:N-1;
p4(m,n)=du(m,n)-du(m-1,n-1);
p=p1+p2+p3+p4;                          %求ρ(x,y)

p(1,1)=dx(1,1)+dy(1,1)+du(1,1);   %边界条件
n=2:N;
p(1,n)=dx(1,n)+dy(1,n)-dy(1,n-1)+du(1,n)+dv(1,n); %边界条件
n=2:N-1;
p(M,n)=-dx(M-1,n)+dy(M,n)-dy(M,n-1)-dv(M-1,n+1)-du(M-1,n-1); %边界条件
m=2:M;
p(m,1)=dx(m,1)-dx(m-1,1)+dy(m,1)-dv(m-1,2)+du(m,1); %边界条件
m=2:M;
p(m,N)=dx(m,N)-dx(m-1,N)-dy(m,N-1)+dv(m,N)-du(m-1,N-1);
m=2:M;
w1(m,:)=p1(m,:)-p1(m-1,:);
m=2:M;
w2(m,:)=p2(m,:)-p2(m-1,:);
m=2:M;
w3(m,:)=p3(m,:)-p3(m-1,:);
m=2:M;
w4(m,:)=p4(m,:)-p4(m-1,:);
wij=sqrt(w1.^2+w2.^2+w3.^2+w4.^2); %采用加权处理，取缠绕相位二阶差分的平方和为权值
k=0.01;
p=(1+k*wij).*p;

pp=dct2(p)+eps;              %对ρ(x,y)作余弦变换
P_unwrap=zeros(M,N);    %求余弦变换后的物体本位Ψ
for m=1:M
    for n=1:N  
     P_unwrap(m,n)=pp(m,n)/(2*cos(pi*(m-1)/M)+2*cos(pi*(n-1)/N)+4*cos(pi*(m-1)/M)*cos(pi*(n-1)/N)-8+eps);
    end
end
P_unwrap(1,1)=pp(1,1);
phase3=idct2(P_unwrap);            %再作反余弦变换得到物体相位  %解包裹出的相位 

% %--------------第五步，从绝对相位中提取物体的三维信息-------------------
len = 100;
d =20;
p=20;
H = p*len*phase3./(p*phase3 +2*pi*d);
figure(20);mesh(H)
title('传统方法深度信息')
%归一化二维矩阵
yi1 = max(max(H));
mi1=H/yi1;
figure(21);imshow(H)
figure(22);surf(mi1)
shading interp
title('归一化深度信息')
toc
