%�Ĳ������������ķ���ʵ��
%����Ĳ����Ϊ�����岽
% 
% --------------��һ������άͼ�ε�ģ��-------------------
% %ģ���ʱ����������Ϊ��λ
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
% set(gca,'xtick',t); %�����仰����ȥ��x��Ŀ̶Ⱥ�����ֵ
% xlabel('x���˵��') %��仰����������������һ��˵��
% set(gca,'ytick',t); %�����仰����ȥ��y��Ŀ̶Ⱥ�����ֵ
% ylabel('y���˵��') %��仰����������������һ��˵�� 
% set(gca,'ztick',t); %�����仰����ȥ��y��Ŀ̶Ⱥ�����ֵ
% zlabel('z���˵��') %��仰����������������һ��˵�� 
% ����������ֵ
xi=1:0.05:7.35;
yi=1:0.05:7.35;
[xi,yi]=meshgrid(xi,yi);
zi=interp2(x,y,temps,xi,yi,'cubic');
yi = max(max(zi));
mi=zi/yi;
x = 1:128;
y = 1:128; 
figure(1);mesh(x,y,zi);
% title('ʵ���������ά����')
% clear indexs;
figure(2);
imshow(mi);
% k=5;
% save_path=['./imgs/',num2str(k),'.bmp'];
% imwrite(mi,save_path,'bmp');%ͼƬ��д�룬��һ����դ
% img = imread(save_path); % ������unit8��(0~255)����
% I1  = im2double(img);    % ��ͼ��ת����double�������ͣ�0~1��
% I2  = double(img)/255;   % uint8ת����double,����ͬim2double
% figure(3);
% surf(I2)

% colorbar  %��ʾ��ɫ��
% figure(4);surf(mi)
% shading interp
% view(0,90) %����ͼxoyƽ�棬xΪ����
% view(-90,0) %����ͼyozƽ�棬yΪ����
% view(0,0) %����ͼxozƽ�棬xΪ����
% figure(5)
% colorbar  %��ʾ��ɫ��
% colormap(jet)  %��ɫ�ķ��ѡ��
%--------------�ڶ��������ҹ�դ������-------------------
x=1:128;
f = 1/20;%һ����դ�ռ�Ƶ��
X = ones(128,1)*x;
fai =2*pi*f.*X;
I00 =1+cos(fai); 
imwrite(I00,'shatter00.bmp','bmp');%ͼƬ��д�룬û�з�����λ

%ϵͳ����������
len =100;%����ĳ�ͫ���ĵ��ο���ľ���
d =20; %ͶӰ���ĵ���������ĵľ���
p =1/f;%ͶӰ��դ�Ŀռ����ڣ�6������
detafai = 2*pi*f*d.*zi/len;%����������ӵ���λ
clear X Y Z I00;

%���ι�դ��д��
I11 = 1 + cos(fai+detafai);%1+cos()
imwrite(I11,'shatter11.bmp','bmp');%ͼƬ��д�룬��һ����դ
I12 = 1 + cos(fai+detafai+pi/2);% 1-sin()
imwrite(I12,'shatter12.bmp','bmp');%ͼƬ��д�룬�ڶ�����դ
I13 = 1 + cos(fai+detafai+pi);% 1-cos()
imwrite(I13,'shatter13.bmp','bmp');%ͼƬ��д�룬��������դ
I14 = 1 + cos(fai+detafai+3*pi/2);%1+sin()
imwrite(I14,'shatter14.bmp','bmp');%ͼƬ��д�룬���ĸ���դ

%ԭʼ��դ��д��
I01 = 1 + cos(fai);%1+cos()
imwrite(I01,'shatter01.bmp','bmp');%ͼƬ��д�룬��һ����դ
I02 = 1 + cos(fai+pi/2);% 1-sin()
imwrite(I02,'shatter02.bmp','bmp');%ͼƬ��д�룬�ڶ�����դ
I03 = 1 + cos(fai+pi);% 1-cos()
imwrite(I03,'shatter03.bmp','bmp');%ͼƬ��д�룬��������դ
I04 = 1 + cos(fai+3*pi/2);%1+sin()
imwrite(I04,'shatter04.bmp','bmp');%ͼƬ��д�룬���ĸ���դ
clear I01 I02 I03 I04 I11 I12 I13 I14;

%--------------��������ͼƬ�Ķ�ȡ�Լ���ʼ���˲�-------------------

h=fspecial('gaussian',9,1);%������˹��ͨ�˲���

%�ο�ͼƬ�����ͼƬ�Ķ�ȡ
%�ο�ͼƬ��ȡ��Ϊ�˻�ȡ�ز���λ
I01=imread('shatter01.bmp','bmp'); %I01=double(I01);
I01=double(filter2(h,double(I01)));
I02=imread('shatter02.bmp','bmp'); %I02=double(I02);
I02=double(filter2(h,double(I02)));
I03=imread('shatter03.bmp','bmp'); %I03=double(I03);
I03=double(filter2(h,double(I03)));
I04=imread('shatter04.bmp','bmp'); 
I04=double(filter2(h,double(I04)));
%���ƹ�դͼƬ�Ķ�ȡ��Ϊ�˻�ȡ���κ����λ
I11=imread('shatter11.bmp','bmp'); %I11=double(I11);
I11=double(filter2(h,double(I11)));
I12=imread('shatter12.bmp','bmp'); %I12=double(I12);
I12=double(filter2(h,double(I12)));
I13=imread('shatter13.bmp','bmp'); %I13=double(I13);
I13=double(filter2(h,double(I13)));
I14=imread('shatter14.bmp','bmp'); I14=double(filter2(h,double(I14)));
[width,height]=size(I11);%���ͼƬ�ĳߴ�

%phase0���ز���λ��phase1�Ǳ������ƺ����λ��phase���������λ������Ҫ���з��������
%phase0���ز���λ��phase1�Ǳ������ƺ����λ��phase���������λ������Ҫ���з��������
phase1 = zeros(128,128);
phase0 = phase1;
phase = phase0;%��λ����ĳ�ʼ��
% ------------------�Ĳ����Ʒ�
for i=1:width
    for j=1:height
        s = abs(atan((I02(i,j)-I04(i,j))/(I01(i,j)-I03(i,j))));
        if(I02(i,j)>=I04(i,j))%sin(fai)<0
            if(I01(i,j)<=I03(i,j))%cos(fai)>0
                phase0(i,j)= -s;%��������
            else
                phase0(i,j)= s - pi;%��������
            end
            
        else%sin(fai)>0
            if(I01(i,j)<=I03(i,j))%cos(fai)>0
                phase0(i,j)= s;%��һ����
            else
                phase0(i,j)= pi - s;%�ڶ�����
            end
        end
    end
end

%���Ʊ��κ����λ����ȡ
for i=1:width
    for j=1:height
        s = abs(atan((I12(i,j)-I14(i,j))/(I11(i,j)-I13(i,j))));
        if(I12(i,j)>=I14(i,j))%sin(fai)<0
            if(I11(i,j)<=I13(i,j))%cos(fai)>0
                phase1(i,j)= -s;%��������
            else
                phase1(i,j)= s - pi;%��������
            end
            
        else%sin(fai)>0
            if(I11(i,j)<=I13(i,j))%cos(fai)>0
                phase1(i,j)= s;%��һ����
            else
                phase1(i,j)= pi - s;%�ڶ�����
            end
        end
    end
end

clear width height s i j detafai fai x y I01 I02 I03 I04 I11 I12 I13 I14
% -----------------���������
% %����������λ��������õ�������λ
P = phase0- phase1;
% % ������С������λ������㷨
[M,N] = size(P);
dx=zeros(M,N); %Ԥ����X������ݶ�
dy=zeros(M,N); %Ԥ����y������ݶ�
du=zeros(M,N); %Ԥ�������¶Խ��߷�����ݶ�
dv=zeros(M,N); %Ԥ�������¶Խ��߷�����ݶ�
m=1:M-1;          %ͨ��ѭ�������ݶ�
dx(m,:)=P(m+1,:)-P(m,:);
dx=dx-pi*floor(dx/pi+0.5);  %���ضϵ��ݶ���������
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
p=zeros(M,N);               %���(x,y)
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
p=p1+p2+p3+p4;                          %���(x,y)

p(1,1)=dx(1,1)+dy(1,1)+du(1,1);   %�߽�����
n=2:N;
p(1,n)=dx(1,n)+dy(1,n)-dy(1,n-1)+du(1,n)+dv(1,n); %�߽�����
n=2:N-1;
p(M,n)=-dx(M-1,n)+dy(M,n)-dy(M,n-1)-dv(M-1,n+1)-du(M-1,n-1); %�߽�����
m=2:M;
p(m,1)=dx(m,1)-dx(m-1,1)+dy(m,1)-dv(m-1,2)+du(m,1); %�߽�����
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
wij=sqrt(w1.^2+w2.^2+w3.^2+w4.^2); %���ü�Ȩ����ȡ������λ���ײ�ֵ�ƽ����ΪȨֵ
k=0.01;
p=(1+k*wij).*p;

pp=dct2(p)+eps;              %�Ԧ�(x,y)�����ұ任
P_unwrap=zeros(M,N);    %�����ұ任������屾λ��
for m=1:M
    for n=1:N  
     P_unwrap(m,n)=pp(m,n)/(2*cos(pi*(m-1)/M)+2*cos(pi*(n-1)/N)+4*cos(pi*(m-1)/M)*cos(pi*(n-1)/N)-8+eps);
    end
end
P_unwrap(1,1)=pp(1,1);
phase3=idct2(P_unwrap);            %���������ұ任�õ�������λ  %�����������λ 

% %--------------���岽���Ӿ�����λ����ȡ�������ά��Ϣ-------------------
len = 100;
d =20;
p=20;
H = p*len*phase3./(p*phase3 +2*pi*d);
figure(20);mesh(H)
title('��ͳ���������Ϣ')
%��һ����ά����
yi1 = max(max(H));
mi1=H/yi1;
figure(21);imshow(H)
figure(22);surf(mi1)
shading interp
title('��һ�������Ϣ')
toc
