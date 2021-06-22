function [U,lambda]=eigen_training(A)

M                   =   size(A,2);
N                   =   size(A,1);
L                   =   A'*A;
% calcolo autovalori di A'A
[vettori,valori]    =   eig(L);
valori              =   diag(valori);

[lambda, ind]    =   sort(valori,'descend');
vettori          =   vettori(:,ind);

%determino gli autovettori di AA'
U                   =   A*vettori;

for i=1:M
    U(:,i)=U(:,i)/norm(U(:,i));
end

