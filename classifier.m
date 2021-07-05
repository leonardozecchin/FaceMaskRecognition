function [WithMask,NoMask,label] = classifier(YT,T,T1,T2,mean1,sigma1,mean2,sigma2)

%Memorizzazione dimensione test
[row,col1] = size(T1);
[row,col2] = size(T2);
[row,col] = size(T);

label = ones(1,col);
label(:,col1+1:col) = 2;

%Creazione delle classi with_mask without_mask
WithMask=[];
NoMask=[];
%z=0; Provo a commentarlo

%Learning del modello generativo di Bayes
for z=1:col
    t = YT(:,z);
    %Calcolo della likehood per ogni punto del dataset di learning
    LK1 = sum(log(normpdf(double(t),double(mean1),double(sigma1'+eps))));
    LK2 = sum(log(normpdf(double(t),double(mean2),double(sigma2'+eps))));
    %Classificazione del punto
    if LK1 >LK2
        WithMask = [WithMask,z];
    else
        NoMask = [NoMask,z];
    end
end

end