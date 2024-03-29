%% Setup - Preparazione del dataset - Tempo : 1 minuto
close all
clear all

% Import delle immagini 
images_dir = 'FaceMaskDataset/Train/WithMask/'; %Immagini di train con la maschera
images_dirNM = 'FaceMaskDataset/Train/WithoutMask/'; %Immagini di train senza maschera
list = dir(strcat(images_dir,'*.png')); %Struttura dati che contiene le informazioni delle immagini con maschera 
listNM = dir(strcat(images_dirNM,'*.png')); %Struttura dati che contiene le informazioni delle immagini senza maschera
M = size(list,1);
M = M + size(listNM,1) %Numero delle immagini insieme
%fact = 0.2; % resizing percentage factor % Non serve più perché facciamo
%il resize in modo diverso
tmp = imresize(imread(strcat(images_dir,'/',list(1).name)),[50 50]); %Resize delle immagini in modo che siano tutte uguali e che non esploda il PC
%tmp1 = imread(strcat(images_dir,'/',list(2).name));
%[r1,c1,ch1] = size(tmp1);
[r,c,ch] = size(tmp); %Dimensioni delle immagini, altezza, larghezza e colore
%label = ([ones(180,1);ones(157,1)*2]);

for i=1:size(list,1) %Trasformazione dei valori delle immagini in un singolo vettore e aggiunta di queste nel vettore TMP
    tmp         =   imresize(imread(strcat(images_dir,'/',list(i).name)),[50 50]);
    tmp1        =   reshape(tmp,r*c*ch,1);                                
    TMP1(:,i)    =   tmp1; 
end

for j=1:size(listNM,1) %Uguale a prima ma vengono aggiunte le immagini senza maschera e in ordine
    tmp2 = imresize(imread(strcat(images_dirNM,'/',listNM(j).name)),[50 50]);
    tmp22        =   reshape(tmp2,r*c*ch,1);
    TMP2(:,j) = tmp22;
end

%% Prima parte LDA - Calcolo della within e between class matrix - Tempo : 12 secondi

%Casting a double
TMP1 = double(TMP1);
TMP2 = double(TMP2);

%Calcolo della media delle 2 classi
Mu1 = mean(TMP1')';
Mu2 = mean(TMP2')';

%Calcolo della media di tutti i punti di entrambe le classi
Mu = (Mu1 + Mu2)./2;

%Calcolo della matrice di covarianza
S1 = cov(TMP1');
S2 = cov(TMP2');

%Calcolo della within-class scatter matrix
Sw = S1 + S2;


%Numero di immagini per ogni classe
N1 = size(TMP1,2);
N2 = size(TMP2,2);

%Calcolo della between-class scatter matrix
SB1 = N1 .* (Mu1-Mu)*(Mu1-Mu)';
SB2 = N2 .* (Mu2-Mu)*(Mu2-Mu)';

%Somma between-class scatter matrix delle 2 classi
SB = SB1 + SB2;

%% Seconda parte LDA - Calcolo della proiezione di LDA - Tempo : 4 minuti e 41 secondi
invSw = inv(Sw);
invSW_by_SB = invSw * SB;

%Calcolo degli autovettori
[V,D] = eig(invSW_by_SB);

%Estrazione dei 2 migliori autovettori della matrice invSW_by_SB
W1 = V(:,1);
W2 = V(:,2);

%Proiezione 
y1_w1 = W1'*TMP1;
y2_w1 = W1'*TMP2;

%% Plotting - Stampa dei punti proiettati - Tempo : 3 secondi
%Stampa dei punti delle 2 classi prima della proiezione di LDA
figure;
scatter(TMP1(1,:),TMP1(2,:),'r');
hold on
scatter(TMP2(1,:),TMP2(2,:),'b');
%figure, scatter(y1_w1,ones(1,N1),[])

%Stampa dei punti proiettati in una dimensione post-LDA
figure;
scatter(y1_w1,ones(1,N1),'r');
hold on
scatter(y2_w1,ones(1,N2),'b');
hold on

%% Plotting aggiuntivo - Stampa bidimensionale delle gaussiane post-LDA - Tempo : 1 secondo

minY = min([min(y1_w1),min(y2_w1)]);
maxY = min([max(y1_w1),max(y2_w1)]);
y_w1 = minY:0.05:maxY;


y1_w1_Mu = mean(y1_w1);
y1_w1_sigma = std(y1_w1);
y1_w1_pdf = mvnpdf(y_w1',y1_w1_Mu, y1_w1_sigma);

y2_w1_Mu = mean(y2_w1);
y2_w1_sigma = std(y2_w1);
y2_w1_pdf = mvnpdf(y_w1',y2_w1_Mu, y2_w1_sigma);


figure;
plot(y1_w1_pdf,'r');
hold on
plot(y2_w1_pdf,'b');
hold on

%% Work in progress

y1_w2 = W2'*TMP1;
y2_w2 = W2'*TMP2;

figure;
scatter(y1_w2,ones(1,N1),'r');
hold on
scatter(y2_w2,ones(1,N2),'b');
hold on



t = -10:500;
line_x1 = t .* W1(1);
line_y1 = t .* W1(1);

line_x2 = t .* W2(1);
line_y2 = t .*W2(1);

plot(line_x1,line_y1,'k-','LineWidth',3);
hold on
plot(line_x2,line_y2,'m-','LineWidth',3);
grid on
