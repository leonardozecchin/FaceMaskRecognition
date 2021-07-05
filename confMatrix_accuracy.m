function [accuracy,precision,recall] = confMatrix_accuracy(label,WithMask,NoMask)
%Calcolo matrice di confusione dateset di test
classif = label.*0;
classif(WithMask)=1;
classif(NoMask)=2;
goodtest = find(classif~=0);
confmat = zeros(2,2); %Matrice di confusione
for i=1:length(goodtest)
    el = goodtest(i);
     confmat(classif(el),label(el))=...
         confmat(classif(el),label(el))+1;
end

num_class = 2;
for i=1:num_class
    precision(i) = confmat(i,i)/(sum(confmat(:,i)));
    recall(i) = confmat(i,i)/(sum(confmat(i,:)));
end

%Calcolo accuratezza con confMatrix dataset di test
accuracy = sum(diag(confmat))/sum(confmat(:));
end