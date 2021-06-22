%SETUP
close all
clear all

images_dir          =   'FaceMaskDataset/Train/WithMask/';
list                =   dir(strcat(images_dir,'*.png'));
M                   =   size(list,1);
fact                =   0.2; % resizing percentage factor
tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[50 50]);
%tmp1 = imread(strcat(images_dir,'/',list(2).name));
%[r1,c1,ch1] = size(tmp1);
[r,c,ch]            =   size(tmp);
%label               =   ([ones(180,1);ones(157,1)*2]);

for i=1:size(list,1)
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[50 50]);
    tmp1        =   reshape(tmp,r*c*ch,1);
                                    
    TMP(:,i)    =   tmp1; 
end

h1=figure; imshow(strcat(images_dir,'/',list(4323).name)); title('Imshow'); 
set(gcf,'Name','Imshow','IntegerHandle','off'); 
colorbar

