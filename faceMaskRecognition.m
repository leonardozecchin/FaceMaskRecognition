%% Setup - Gestione delle immagini del dataset - Tempo : 24 secondi

close all
clear all

%Import delle immagini di training
images_dir = 'FaceMaskDataset/Train/WithMask/'; 
images_dirNM = 'FaceMaskDataset/Train/WithoutMask/';
list = dir(strcat(images_dir,'*.png')); %Struttura dati che contiene le informazioni delle immagini con maschera 
listNM = dir(strcat(images_dirNM,'*.png')); %Struttura dati che contiene le informazioni delle immagini senza maschera
M = size(list,1);
M = M + size(listNM,1) %Numero delle immagini insieme
tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[30 30]); %Resize delle immagini in modo che siano tutte uguali e che non esploda il PC
[r,c,ch] = size(tmp); %Dimensioni delle immagini, altezza, larghezza e colore

%Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
for i=1:size(list,1)
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[30 30]);
    tmp1        =   reshape(tmp,r*c*ch,1);                                
    TMP1(:,i)    =   tmp1; 
end

for j=1:size(listNM,1)
    tmp2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[30 30]);
    tmp22        =   reshape(tmp2,r*c*ch,1);
    TMP2(:,j) = tmp22;
end

%Insieme di tutti i valori delle immagini di training
TMP = [TMP1,TMP2];

%Import delle immagini di testing
images_dirTest = 'FaceMaskDataset/Test/WithMask/'; %Immagini di train con la maschera
images_dirTestNM = 'FaceMaskDataset/Test/WithoutMask/'; %Immagini di train senza maschera
listTest = dir(strcat(images_dirTest,'*.png')); %Struttura dati che contiene le informazioni delle immagini con maschera 
listTestNM = dir(strcat(images_dirTestNM,'*.png')); %Struttura dati che contiene le informazioni delle immagini senza maschera 

MT = size(listTest,1);
MT = MT + size(listTestNM,1) %Numero delle immagini insieme

%Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
for i=1:size(listTest,1)
    test         =   imresize(imread(strcat(images_dirTest,'/',listTest(i).name)),[30 30]); %Resize delle immagini in modo che siano tutte uguali e che non esploda il PC
    [r,c,ch] = size(test); %Dimensioni delle immagini, altezza, larghezza e colore

    test1        =   reshape(test,r*c*ch,1);                                
    Test1(:,i)    =   test1; 
end

for j=1:size(listTestNM,1)
    test2 = imresize(imread(strcat(images_dirTestNM,'/',listTestNM(j).name)),[30 30]);
    [r,c,ch] = size(test2);
    test22        =   reshape(test2,r*c*ch,1);
    Test2(:,j) = test22;
end

%Insieme di tutti i valori delle immagini di testing
Test = [Test1,Test2];
Test = double(Test);

%% Prima parte LDA - Calcolo matrici within e between class - Tempo : 3 minuti 46 secondi

TMP = double(TMP);
l = reshape(repmat([1:2],5000,1),M,1); %Etichettatura delle prime 5000 immagini come immagini con mascherina e le seconde 5000 come senza mascherina
[d,N] = size(TMP); %Dimensione della matrice che contiene i punti
K = max(l);

% 1. determino le classi Ck
for k = 1:K
    a = find (l == k); %Prende le posizioni delle immagini prima che hanno etichetta 1 in l e poi 2
    Ck{k} = TMP(:,a); %Salva nella struttura Ck (chiamata cella) tutte le X che appartengono prima alla classe 1 poi alla 2
end

% 2. determino le medie
for k = 1:K
    mk{k} = mean(Ck{k},2); %Media dei valori su 1 e media dei valori su 2
end

% 3. determino la numerosità della classe
for k = 1:K
    [d, Nk(k)] = size(Ck{k}); %Determino il numero di dati per ogni classe
end

% 4. determino le within class scatter
for k = 1:K
    S{k} = 0; %Inizializza la cella a 0
    for i = 1:Nk(k) %Nk è la size dei dati per ogni classe (numero di immagini (5000 per classe))
        S{k} = S{k} + (Ck{k}(:,i)-mk{k})*(Ck{k}(:,i)-mk{k})'; %Calcola la matrice di Scatter ovvero la matrice di covarianza utilizzata per l'estrazione delle feature
    end
    S{k} = S{k}./Nk(k); %Divisione per il numero di immagini, credo serva per normalizzare la matrice di scatter
end
Swx = 0;
for k = 1:K
    Swx = Swx + S{k}; %Within class scatter matrix
end

% 5. determino la between class covariance
% 5.1 determino la media totale
m = mean(TMP,2);
Sbx = 0;
for k=1:K
    Sbx = Sbx + Nk(k)*((mk{k} - m)*(mk{k} - m)'); %Calcolo della between class scatter matrix
end
Sbx = Sbx/K; %Normalizzazione della between class scatter matrix sul numero di classi

MA = inv(Swx)*Sbx; %Applicazione della LDA projection

%% Seconda parte LDA - Estrazione dell'autovettore e proiezione dei punti - Tempo : 9 secondi

% eigenvalues/eigenvectors
[V,D] = eig(MA); %Estazione degli autovettori dalla porecedente matrice

%Sort degli auvalori della matrice MA
D=diag(D);
[D,ind]=sort(D,'descend');
V = V(:,ind);

% 5: transform matrix
A = V(:,1); %Scelgo il singolo migliore autovettore su cui proiettare i punti
% 6: transformation
Y = A'*TMP; %Proiezione vera e propria

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

%figure;
%scatter(Y1,ones(1,5000),[],'r');
%hold on 
%scatter(Y(:,5001:10000),ones(1,5000),[],'b');
%hold on 

%scatter 1D di Y1 e Y2 insieme dove Y1 e' blue mentre Y2 e' rosso
figure;
scatter(Y1,zeros(5000,1),10,'b');
hold on
scatter(Y2,zeros(5000,1),10,'r');

figure;
scatter(Y1,normpdf(Y1,mean1,sigma1)*100,10,'b');
hold on
scatter(Y2,normpdf(Y2,mean2,sigma2)*100,10,'r');


%% Testing - Classificazione dei punti di testing - Tempo : 1 secondo

%Proiezione dei dati di test usando LDA
YT = A'*Test;

%Memorizzazione dimensione test
[row1,col1] = size(Test1);
[row2,col2] = size(Test2);
[row,col] = size(Test);

labelTest = ones(1,col);
labelTest(:,col1+1:col) = 2;

%Creazione delle classi with_mask without_mask
C1=[];
C2=[];
%z=0; Provo a commentarlo

%Learning del modello generativo di Bayes
for z=1:col
    t = YT(:,z);
    %Calcolo della likehood per ogni punto del dataset di learning
    LK1 = sum(log(normpdf(double(t),double(mean1),double(sigma1'+eps))));
    LK2 = sum(log(normpdf(double(t),double(mean2),double(sigma2'+eps))));
    %Classificazione del punto
    if LK1 >LK2
        C1 = [C1,z];
    else
        C2 = [C2,z];
    end
end

%% Prima parte accuratezza - Calcolo accuracy - Tempo : 1 secondo

%Calcolo accuratezza classificazione dataset di test
countc = 0;
count = length(labelTest); % subtract the training elements
for i=C1;
    if labelTest(i)==1
        countc = countc + 1;
    end
end

for i=C2;
    if labelTest(i)==2
        countc = countc + 1;
    end
end
%Calcolo accuracy dataset di test
accuracy = countc/count;

%% Seconda parte accurattezza - Calcolo matrice di confusione - Tempo : 1 secondo

%Calcolo matrice di confusione dateset di test
classif = labelTest.*0;
classif(C1)=1;
classif(C2)=2;
goodtest = find(classif~=0);
confmat = zeros(2,2); %Matrice di confusione
for i=1:length(goodtest)
    el = goodtest(i);
     confmat(classif(el),labelTest(el))=...
         confmat(classif(el),labelTest(el))+1;
end

num_class = 2;
for i=1:num_class
    precision(i) = confmat(i,i)/(sum(confmat(:,i)));
    recall(i) = confmat(i,i)/(sum(confmat(i,:)));
end
%Calcolo accuratezza con confMatrix dataset di test
accuracy = sum(diag(confmat))/sum(confmat(:))

%Creo due array che contengono i falsi positivi in C1 e C2
fakeC1= [];
for i=1:length(C1)
    if C1(i)>483
        fakeC1 = [fakeC1,C1(i)];
    end
end
    
fakeC2= [];
for i=1:length(C2)
    if C2(i)<484
        fakeC2 = [fakeC2,C2(i)];
    end
end
   

%qui cambio la numerazione dei falsi positivi in C1 per avere i numeri
%corretti corrispondenti alla cartella delle immagini di NoMask
false_positive1 = [];
for i=1:length(fakeC1)
    if fakeC1(i)>483
        value = col - fakeC1(i);
        false_positive1 = [false_positive1,col2-value];
    end
end

%% Plotting - Stampa immagini precise del data set

%Stampare le immagini richieste dell'insieme
img = imread(strcat(images_dirTest,'/',listTest(483).name));
imshow(img);
img = imread(strcat(images_dirTestNM,'/',listTestNM(27).name));
imshow(img);

%% Plotting - Work in progress

%function [] = plot_result_matrix(precision,recall, accuracy,method)
method = 'Single gaussian estimated parameters';
symbol_max = 'max';
symbol_min = 'min';

features_vec = ([1:5] *10).^2 * 3;

figure('NumberTitle', 'off', 'Name', method);
subplot(2,1,1)

%precision = [result_matrix{1,1}.precision result_matrix{1,2}.precision result_matrix{1,3}.precision result_matrix{1,4}.precision result_matrix{1,5}.precision];
%recall = [result_matrix{1,1}.recall result_matrix{1,2}.recall result_matrix{1,3}.recall result_matrix{1,4}.recall result_matrix{1,5}.recall];
%accuracy = [result_matrix{1,1}.accuracy result_matrix{1,2}.accuracy result_matrix{1,3}.accuracy result_matrix{1,4}.accuracy result_matrix{1,5}.accuracy];

plot(features_vec,precision);

[y,x] = max(precision);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(precision == min(min(precision)));
text((x*10).^2*3,min(min(precision)),symbol_min);

hold on;
plot(features_vec,recall);

[y,x] = max(recall);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(recall == min(min(recall)));
text((x*10).^2*3,min(min(recall)),symbol_min);

hold on;
plot(features_vec,accuracy);

[y,x] = max(accuracy);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(accuracy == min(min(accuracy)));
text((x*10).^2*3,min(min(accuracy)),symbol_min);


title('test set')
xlabel('features')

legend({'precision','recall','accuracy'},'Location','southwest')

subplot(2,1,2)

%precision = [a{2,1}.precision a{2,2}.precision a{2,3}.precision a{2,4}.precision a{2,5}.precision];
%recall = [a{2,1}.recall a{2,2}.recall a{2,3}.recall a{2,4}.recall a{2,5}.recall];
%accuracy = [a{2,1}.accuracy a{2,2}.accuracy a{2,3}.accuracy a{2,4}.accuracy a{2,5}.accuracy];

plot(features_vec,precision);

[y,x] = max(precision);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(precision == min(min(precision)));
text((x*10).^2*3,min(min(precision)),symbol_min);

hold on;
plot(features_vec,recall);

[y,x] = max(recall);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(recall == min(min(recall)));
text((x*10).^2*3,min(min(recall)),symbol_min);

hold on;
plot(features_vec,accuracy);

[y,x] = max(accuracy);
text((x*10).^2*3,y,symbol_max);

[~,x] = find(accuracy == min(min(accuracy)));
text((x*10).^2*3,min(min(accuracy)),symbol_min);

title('validation set')
xlabel('features')

legend({'precision','recall','accuracy'},'Location','southwest')
%end

