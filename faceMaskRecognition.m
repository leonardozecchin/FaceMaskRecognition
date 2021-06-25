%%SETUP immagini

close all
clear all

%Import delle immagini 
images_dir = 'archive/FaceMaskDataset/Train/WithMask/'; %Immagini di train con la maschera
images_dirNM = 'archive/FaceMaskDataset/Train/WithoutMask/'; %Immagini di train senza maschera
list = dir(strcat(images_dir,'*.png')); %Struttura dati che contiene le informazioni delle immagini con maschera 
listNM = dir(strcat(images_dirNM,'*.png')); %Struttura dati che contiene le informazioni delle immagini senza maschera
M = size(list,1);
M = M + size(listNM,1) %Numero delle immagini insieme
%fact = 0.2; % resizing percentage factor % Non serve più perché facciamo
%il resize in modo diverso
tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[30 30]); %Resize delle immagini in modo che siano tutte uguali e che non esploda il PC
%tmp1 = imread(strcat(images_dir,'/',list(2).name));
%[r1,c1,ch1] = size(tmp1);
[r,c,ch] = size(tmp); %Dimensioni delle immagini, altezza, larghezza e colore
%label = ([ones(180,1);ones(157,1)*2]);

for i=1:size(list,1) %Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[30 30]);
    tmp1        =   reshape(tmp,r*c*ch,1);                                
    TMP1(:,i)    =   tmp1; 
end

for j=1:size(listNM,1) %Uguale a prima ma vengono aggiunte le immagini senza maschera e in ordine
    tmp2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[30 30]);
    tmp22        =   reshape(tmp2,r*c*ch,1);
    TMP2(:,j) = tmp22;
end

TMP = [TMP1,TMP2]; %Insieme di tutti i valori delle immagini

%% Prima parte LDA - Setup e procedura PCA

%media = mean(TMP,2); %Media del TMP

%AA(:,:) = TMP-repmat(media,1,M); %Sottrazione della media ai punti delle immagini
%[U,lambda] = eigen_training(AA); %Calcolo degli autovettori sulla matrice a cui è stata sottratta la media
%T = 500; % Numero dei migliori autovettori da tenere, questo implica anche la dimensione dei dati proiettati sul nuovo asse
%Prima si fà PCA e poi LDA
%X = U(:,1:T)'*AA; %Proiezione dei punti senza media sugli autovettori migliori (QUESTA E' LA PROIZIONE DI PCA);
 % numero classi in gioco;


%% Seconda parte LDA - Calcolo matrici within e between class
TMP = double(TMP); %Casting a double
l = reshape(repmat([1:2],5000,1),M,1); %Etichettatura delle prime 5000 immagini come immagini con mascherina e le seconde 5000 come senza mascherina
%l = reshape(repmat([1:40],10,1),400,1);
[d,N] = size(TMP); %Dimensione della matrice che contiene i punti proiettati seguendo PCA, quindi dimensione di X
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

%% Terza parte LDA - Estrazione dell'autovettore e proiezione dei punti

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

sigma1 = std(Y1);
mean1 = mean(Y1);

sigma2 = std(Y2);
mean2 = mean(Y2);

%% Plotting dei dati proiettati
% 7: plot
%figure;
%scatter(Y,ones(1,N),[],l);

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


%% Learning-Ricerca della Maximum Likelihood 

images_dirTest = 'archive/FaceMaskDataset/Test/WithMask/'; %Immagini di train con la maschera
images_dirTestNM = 'archive/FaceMaskDataset/Test/WithoutMask/'; %Immagini di train senza maschera
listTest = dir(strcat(images_dirTest,'*.png')); %Struttura dati che contiene le informazioni delle immagini con maschera 
listTestNM = dir(strcat(images_dirTestNM,'*.png'));

MT = size(listTest,1);
MT = MT + size(listTestNM,1) %Numero delle immagini insieme
tmpT = imresize(imread(strcat(images_dirTest,'/',listTest(1).name)),[30 30]); %Resize delle immagini in modo che siano tutte uguali e che non esploda il PC
[r,c,ch] = size(tmpT); %Dimensioni delle immagini, altezza, larghezza e colore

for i=1:size(listTest,1) %Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
    test         =   imresize(imread(strcat(images_dirTest,'/',listTest(i).name)),[30 30]);
    test1        =   reshape(test,r*c*ch,1);                                
    Test1(:,i)    =   test1; 
end

for j=1:size(listNM,1) %Uguale a prima ma vengono aggiunte le immagini senza maschera e in ordine
    test2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[30 30]);
    test22        =   reshape(test2,r*c*ch,1);
    Test2(:,j) = test22;
end


Test = [Test1,Test2];
Test = double(Test);

% proiezione in una dimensione della matrice delle immagini di Test
YT = A'*Test;

[row1,col1] = size(Test1);
[row2,col2] = size(Test2);
[row,col] = size(Test);

labelTest = ones(1,col);
labelTest(:,col1+1:col) = 2;

C1=[];
C2=[];

z=0;
for z=1:col
    t = YT(:,z);
    LK1 = sum(log(normpdf(double(t),double(mean1),double(sigma1'+eps))));
    LK2 = sum(log(normpdf(double(t),double(mean2),double(sigma2'+eps))));
    if LK1 >LK2
        C1 = [C1,z];
    else
        C2 = [C2,z];
    end
    
end





