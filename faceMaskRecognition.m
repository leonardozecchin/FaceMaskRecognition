%SETUP
close all
clear all

images_dir          =   'archive/FaceMaskDataset/Train/WithMask/';
images_dirNM          =   'archive/FaceMaskDataset/Train/WithoutMask/';
list                =   dir(strcat(images_dir,'*.png'));
listNM = dir(strcat(images_dirNM,'*.png'));
M                   =   size(list,1);
M = M + size(listNM,1)
fact                =   0.2; % resizing percentage factor
tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[50 50]);
%tmp1 = imread(strcat(images_dir,'/',list(2).name));
%[r1,c1,ch1] = size(tmp1);
[r,c,ch]            =   size(tmp);
%label               =   ([ones(180,1);ones(157,1)*2]);

for i=1:size(list,1)
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[50 50]);
    tmp1        =   reshape(tmp,r*c*ch,1);                                
    TMP1(:,i)    =   tmp1; 
end

for j=1:size(listNM,1)
    tmp2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[50 50]);
    tmp22        =   reshape(tmp2,r*c*ch,1);
    TMP2(:,j) = tmp22;
end

TMP = [TMP1,TMP2];


%Check
s= TMP2(:,32);
sappde = TMP(:,8000);
sappdino = imresize(imread(strcat(images_dirNM,'/',listNM(3000).name)),[50 50]);
sapp = reshape(sappdino,r*c*ch,1); 
%It must fail
if isequal(sappde,sapp)
    fprintf("PEJNE\n");
end


h1=figure; imshow(strcat(images_dirNM,'/',listNM(1).name)); title('Imshow'); 
set(gcf,'Name','Imshow','IntegerHandle','off'); 
colorbar



TMP                     =   double(TMP);
media                   =   mean(TMP,2);
AA(:,:)                 =   TMP-repmat(media,1,M);
[U,lambda]              =   eigen_training(AA);
T           =   1000; % sara' da cambiare
X           =   U(:,1:T)'*AA; % projection;
%l           =   reshape(repmat([1:40],10,1),400,1);

l = reshape(repmat([1:2],5000,1),M,1);
[d,N] = size(X);
K = max(l); % numero classi in gioco;

% 1. determino le classi Ck
for k = 1:K
    a = find (l == k);
    Ck{k} = X(:,a);
end

% 2. determino le medie
for k = 1:K
    mk{k} = mean(Ck{k},2);
end

% 3. determino la numerosità della classe
for k = 1:K
    [d, Nk(k)] = size(Ck{k});
end

% 4. determino le within class scatter
for k = 1:K
    S{k} = 0;
    for i = 1:Nk(k)
        S{k} = S{k} + (Ck{k}(:,i)-mk{k})*(Ck{k}(:,i)-mk{k})';
    end
    S{k} = S{k}./Nk(k);
end
Swx = 0;
for k = 1:K
    Swx = Swx + S{k};
end

% 5. determino la between class covariance
% 5.1 determino la media totale
m = mean(X,2);
Sbx = 0;
for k=1:K
    Sbx = Sbx + Nk(k)*((mk{k} - m)*(mk{k} - m)');
end
Sbx = Sbx/K;

MA = inv(Swx)*Sbx;

% eigenvalues/eigenvectors
[V,D] = eig(MA);

% 5: transform matrix
A = V(:,1);

% 6: transformation
Y = A'*X;

% 7: plot
figure, scatter(Y,ones(1,N),[],l)
colormap jet
for i=1:M
    text(Y,num2str(l(i)))
end


figure, scatter(TMP(1,:),TMP(2,:),[],l)
for i=1:M
    text(TMP(1,i),TMP(2,i),num2str(l(i)))
end






