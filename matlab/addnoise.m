for k=1:1:20
    imgs=['./data/imgs/',num2str(k),'.bmp'];
    img=imread(imgs);
%     figure(1);
%     imshow(img);
    img_gaussian=imnoise(img, 'gaussian' , 0, 0.03 ); %��ֵΪ0������Ϊ0.02
%     figure(2);
%     imshow(img_gaussian);
    save_masks=['./noisedata/imgs/',num2str(k),'.bmp'];
    imwrite(img_gaussian,save_masks,'bmp');%����ͼ
end