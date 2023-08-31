for k=1:4:6000
    x = 1:128;
    y = 1:128; 
    [X,Y]=meshgrid(x,y);
    MM=rand()*20;
    NN=0.02+(0.06-0.02) * rand();
    zi=MM*(sin(NN*pi*X)+cos(NN*pi*Y))+30;
    % figure(1);
    % surf(zi);
    % % 图二制作imgs（连续抖动条纹）
    x=1:128;
    f = 1/20;%一个光栅空间频率
    X = ones(128,1)*x;
    fai =2*pi*f.*X;
    len =100;%相机的出瞳中心到参考面的距离
    d =20; %投影中心到照相机中心的距离
    p =1/f;%投影光栅的空间周期，6个像素
    detafai = 2*pi*f*d.*zi/len;%物体表面增加的相位
    ni = 1 + cos(fai+detafai);%1+cos()单通道图片
    % figure(2);
    % imshow(ni);
    save_imgs=['./data/imgs/',num2str(k),'.bmp'];
    imwrite(ni,save_imgs,'bmp');%深度图

    H = p*len*detafai./(p*detafai +2*pi*d);
    yi = max(max(H));
    mi=H/yi;
    save_masks=['./data/masks/',num2str(k),'.bmp'];
    imwrite(mi,save_masks,'bmp');%条纹图
end


