%% Setup - Gestione delle immagini del dataset - Tempo : 24 secondi

close all
clear all

%numero di pixel per ridimensionare le immagini
pixel_num = 10;
%Import delle immagini di training
images_dir = 'archive/FaceMaskDataset/Train/WithMask/'; 
images_dirNM = 'archive/FaceMaskDataset/Train/WithoutMask/';
list = dir(strcat(images_dir,'*.png')); 
listNM = dir(strcat(images_dirNM,'*.png'));

tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[pixel_num pixel_num]); 
[r,c,ch] = size(tmp); %Dimensioni delle immagini, altezza, larghezza e colore

%Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
for i=1:size(list,1)
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[pixel_num pixel_num]);
    tmp1        =   reshape(tmp,r*c*ch,1);                                
    TMP1(:,i)    =   tmp1; 
end

for j=1:size(listNM,1)
    tmp2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[pixel_num pixel_num]);
    tmp22        =   reshape(tmp2,r*c*ch,1);
    TMP2(:,j) = tmp22;
end

%Insieme di tutti i valori delle immagini di training
TMP = [TMP1,TMP2];

%Import delle immagini di testing
images_dirTest = 'archive/FaceMaskDataset/Test/WithMask/'; 
images_dirTestNM = 'archive/FaceMaskDataset/Test/WithoutMask/'; 
listTest = dir(strcat(images_dirTest,'*.png')); 
listTestNM = dir(strcat(images_dirTestNM,'*.png')); 

%Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
for i=1:size(listTest,1)
    test         =   imresize(imread(strcat(images_dirTest,'/',listTest(i).name)),[pixel_num pixel_num]); 
    [r,c,ch] = size(test); %Dimensioni delle immagini, altezza, larghezza e colore

    test1        =   reshape(test,r*c*ch,1);                                
    Test1(:,i)    =   test1; 
end

for j=1:size(listTestNM,1)
    test2 = imresize(imread(strcat(images_dirTestNM,'/',listTestNM(j).name)),[pixel_num pixel_num]);
    [r,c,ch] = size(test2);
    test22        =   reshape(test2,r*c*ch,1);
    Test2(:,j) = test22;
end

%Insieme di tutti i valori delle immagini di testing
Test = [Test1,Test2];
Test = double(Test);

%Import delle immagini di validation
images_dirVal = 'archive/FaceMaskDataset/Validation/WithMask/';
images_dirValNM = 'archive/FaceMaskDataset/Validation/WithoutMask/'; 
listVal = dir(strcat(images_dirVal,'*.png')); 
listValNM = dir(strcat(images_dirValNM,'*.png')); 

%Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
for i=1:size(listVal,1)
    val = imresize(imread(strcat(images_dirVal,'/',listVal(i).name)),[pixel_num pixel_num]); 
    [r,c,ch] = size(val); %Dimensioni delle immagini, altezza, larghezza e colore
    val1 = reshape(val,r*c*ch,1);                                
    Val1(:,i) = val1; 
end

for j=1:size(listValNM,1)
    val2 = imresize(imread(strcat(images_dirValNM,'/',listValNM(j).name)),[pixel_num pixel_num]);
    [r,c,ch] = size(val2);
    val22 = reshape(val2,r*c*ch,1);
    Val2(:,j) = val22;
end

%Insieme di tutti i valori delle immagini di valuation
Val = [Val1,Val2];
Val = double(Val);

%% Prima parte LDA -  Tempo : 3 minuti 46 secondi

M = size(list,1);
M = M + size(listNM,1) %Numero delle immagini insieme

TMP = double(TMP);
l = reshape(repmat([1:2],5000,1),M,1); %Etichettatura delle prime 5000 immagini come immagini con mascherina e le seconde 5000 come senza mascherina
[d,N] = size(TMP); %Dimensione della matrice che contiene i punti
K = max(l);

% Assegnazione delle classi
for k = 1:K
    a = find(l == k); 
    Ck{k} = TMP(:,a); 
end

%Calcolo delle medie per ogni classe
for k = 1:K
    mk{k} = mean(Ck{k},2); 
end

%Determino il numero di dati per ogni classe
for k = 1:K
    [d, Nk(k)] = size(Ck{k}); 
end

%Calcolo della within class scatter
for k = 1:K
    S{k} = 0; 
    for i = 1:Nk(k) 
        S{k} = S{k} + (Ck{k}(:,i)-mk{k})*(Ck{k}(:,i)-mk{k})'; %Calcola la matrice di Scatter ovvero la matrice di covarianza utilizzata per l'estrazione delle feature
    end
    S{k} = S{k}./Nk(k);
end
Swx = 0;
for k = 1:K
    Swx = Swx + S{k}; %Within class scatter matrix
end

%Calcolo della between class covariance
m = mean(TMP,2);
Sbx = 0;
for k=1:K
    Sbx = Sbx + Nk(k)*((mk{k} - m)*(mk{k} - m)'); 
end
Sbx = Sbx/K; %Normalizzazione della between class scatter matrix sul numero di classi

%Applicazione della LDA projection
MA = inv(Swx)*Sbx; 

%% Seconda parte LDA - Estrazione dell'autovettore e proiezione dei punti - Tempo : 9 secondi

%Estazione degli autovettori dalla porecedente matrice
[V,D] = eig(MA); 

%Sorting degli autovalori della matrice MA
D=diag(D);
[D,ind]=sort(D,'descend');
V = V(:,ind);

 %Scelgo il singolo migliore autovettore su cui proiettare i punti
A = V(:,1);
 %Proiezione
Y = A'*TMP;

%Divido Y nelle due classi
Y1 = Y(:,1:5000);
Y2 = Y(:,5001:10000);
%% Modello generativo - Stima dei parametri Gaussiani

%Calcolo parametri classe WithMask
sigma1 = std(Y1);
mean1 = mean(Y1);

%Calcolo parametri classe NoMask
sigma2 = std(Y2);
mean2 = mean(Y2);

%% Plotting - Stampa dati prima e dopo LDA - Tempo : 3 secondi

%scatter 2D del TMP
figure;
scatter(TMP1(1,:),TMP1(2,:),'b');
hold on
scatter(TMP2(1,:),TMP2(2,:),'r');

%scatter 3D del TMP
figure;
scatter3(TMP1(1,:),TMP1(2,:),TMP1(3,:),'b');
hold on
scatter3(TMP2(1,:),TMP2(2,:),TMP2(3,:),'r');

%scatter 1D di Y1 e Y2 insieme dove Y1 e' blue mentre Y2 e' rosso
figure;
scatter(Y1,zeros(5000,1),10,'b');
hold on
scatter(Y2,zeros(5000,1),10,'r');

%plot della distribuzione Gaussiana di Y
figure;
scatter(Y1,normpdf(Y1,mean1,sigma1)*100,10,'b');
hold on
scatter(Y2,normpdf(Y2,mean2,sigma2)*100,10,'r');

%% Testing - Classificazione dei punti di testing - Tempo : 1 secondo

%Proiezione dei dati di test usando LDA
YT = A'*Test;

[WithMask,NoMask,labelTest] = classifier(YT,Test,Test1,Test2,mean1,sigma1,mean2,sigma2);


%% Classificazione delle immagine nella cartella Validation - Tempo : 1 secondo
 
%Proiezione dei dati di test usando LDA
YV = A'*Val;

[WithMaskVal,NoMaskVal,labelVal] = classifier(YV,Val,Val1,Val2,mean1,sigma1,mean2,sigma2);



%% Prima parte accuratezza - Calcolo accuracy - Tempo : 1 secondo

accuracyTest = simple_acccurancy(labelTest,WithMask,NoMask);

accuracyValidation = simple_acccurancy(labelVal,WithMaskVal,NoMaskVal);

%% Seconda parte accurattezza - Calcolo matrice di confusione - Tempo : 1 secondo

[accuracyConfMTest,precisionTest,recallTest] = confMatrix_accuracy(labelTest,WithMask,NoMask);

[accuracyConfMVal,precisionVal,recallVal] = confMatrix_accuracy(labelVal,WithMaskVal,NoMaskVal);

%% Plotting - Stampa immagini precise del data set di Training

%Creo due array che contengono i falsi positivi in WithMask e NoMask
fakeWithMask= [];
for i=1:length(WithMask)
    if WithMask(i)>483
        fakeWithMask = [fakeWithMask,WithMask(i)];
    end
end
    
fakeNoMask= [];
for i=1:length(NoMask)
    if NoMask(i)<484
        fakeNoMask = [fakeNoMask,NoMask(i)];
    end
end
   
%qui cambio la numerazione dei falsi positivi in WithMask per avere i numeri
%corretti corrispondenti alla cartella delle immagini di NoMask
false_positive1 = [];
for i=1:length(fakeWithMask)
    if fakeWithMask(i)>483
        value = col - fakeWithMask(i);
        false_positive1 = [false_positive1,col2-value];
    end
end

%Stampare le immagini richieste dell'insieme
img = imread(strcat(images_dirTest,'/',listTest(483).name));
imshow(img);
img = imread(strcat(images_dirTestNM,'/',listTestNM(27).name));
imshow(img);





