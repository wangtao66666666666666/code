tic;
for k=4:4:12500
% for k=1:1:12500
    x=1:8;
    y=1:8;
    temps=zeros(8,8)
    for i=1:1:8
        for j=1:1:8
            temps(i,j)=rand()*80
        end
    end
    xi=1:0.05:7.35;
    yi=1:0.05:7.35;
    [xi,yi]=meshgrid(xi,yi);
    zi=interp2(x,y,temps,xi,yi,'cubic');
%     figure(1);
%     surf(zi);
%     shading interp
%     figure(1);
%     surf(zi);
    
    x=1:128;
    f = 1/20;%һ����դ�ռ�Ƶ��
    X = ones(128,1)*x;
    fai =2*pi*f.*X;
    len =100;%����ĳ�ͫ���ĵ��ο���ľ���
    d =20; %ͶӰ���ĵ���������ĵľ���
    p=20;
    detafai = 2*pi*f*d.*zi/len;%����������ӵ���λ
    ni = 1 + cos(fai+detafai);%1+cos()��ͨ��ͼƬ
%     save_imgs=['./data/imgs/',num2str(k),'.bmp'];
    save_imgs=['./data/imgs/',num2str(k),'.bmp'];
    imwrite(ni,save_imgs,'bmp');%���ͼ
%     figure(2);
%     imshow(ni);
    
    H = p*len*detafai./(p*detafai +2*pi*d);
% %     figure(11);
% %     surf(H);
    yi = max(max(H));
    mi=H/yi;%��ͨ��ͼƬ 
%     figure(13);
%     imshow(mi);
%     save_masks=['./data/masks/',num2str(k),'.bmp'];
    save_masks=['./data/masks/',num2str(k),'.bmp'];
    imwrite(mi,save_masks,'bmp');%����ͼ
end
toc