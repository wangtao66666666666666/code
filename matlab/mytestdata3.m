for k=4:4:6000
    x = 1:128;
    y = 1:128; 
    [X,Y]=meshgrid(x,y);
    a=rand()*5;
    b=rand()*5;
    zi=(((X-64).^2)/a+((Y-64).^2)/b)/60; 
%     figure(1);
%     surf(zi);
    % % ͼ������imgs�������������ƣ�
    x=1:128;
    f = 1/20;%һ����դ�ռ�Ƶ��
    X = ones(128,1)*x;
    fai =2*pi*f.*X;
    len =100;%����ĳ�ͫ���ĵ��ο���ľ���
    d =20; %ͶӰ���ĵ���������ĵľ���
    p =1/f;%ͶӰ��դ�Ŀռ����ڣ�6������
    detafai = 2*pi*f*d.*zi/len;%����������ӵ���λ
    ni = 1 + cos(fai+detafai);%1+cos()��ͨ��ͼƬ
%     figure(2);
%     imshow(ni);
    save_imgs=['./data/imgs/',num2str(k),'.bmp'];
    imwrite(ni,save_imgs,'bmp');%���ͼ
    
    H = p*len*detafai./(p*detafai +2*pi*d);
    yi = max(max(H));
    mi=H/yi;
%     figure(3);
%     imshow(mi);
    save_masks=['./data/masks/',num2str(k),'.bmp'];
    imwrite(mi,save_masks,'bmp');%����ͼ
end