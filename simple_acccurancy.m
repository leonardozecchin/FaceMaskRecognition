function accuracy = simple_acccurancy(label,WithMask,NoMask)

%Calcolo accuratezza classificazione dataset di test
countc = 0;
count = length(label); % subtract the training elements
for i=WithMask
    if label(i)==1
        countc = countc + 1;
    end
end

for i=NoMask
    if label(i)==2
        countc = countc + 1;
    end
end
%Calcolo accuracy dataset di test
accuracy = countc/count;
end