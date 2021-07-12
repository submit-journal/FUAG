function [W, U, V, P] = un_Fast_parfor_0708(X,c,opt)
% Input:
%     X: dataset  n * d, each row is a point.
%     c: the number of classes.
%     anchor_num: the number of anchor points in each class.
%     k: the number of neighbor anchor points to connect.
%     lambda and gamma are the parameters.
%     dim: the reduced dimensions.
% Output:
%     W: transformation matrix
%     obj_value: objective of function
%     U: index matrix of data point
%     V: index matrix of anchor point
%     P: similarity matrix
zr = 10e-11;
anchor_num = opt(1);
k = opt(2);
lambda = opt(3);
gamma = opt(4);
dim = opt(5);

count = 30;
[n,~]= size(X);

H = eye(n) - 1/n * ones(n);
St = X' * H * X;
St = max(St,St');

m = c * anchor_num;
A_p = zeros(n+m,n+m);
L_p = zeros(n+m,n+m);
% S_t
%% initialize P and F in original space
[~,P] = construct_unAG(X,m,k);
P = P+eps; % in order to avoid no anchor point connected
P = sparse(P);
p1 = sum(P,2);
D1p = spdiags(1./sqrt(p1),0,n,n);
p2 = sum(P,1);
D2p = spdiags(1./sqrt(p2'),0,m,m);
P1 = D1p*P*D2p;
SS2 = P1'*P1;
SS2 = full(SS2); 
% automatically determine the cluster number
[V, ev0, ev]=eig1(SS2,m);
V = V(:,1:c);
U=(P1*V)./(ones(n,1)*sqrt(ev0(1:c)'));
U = sqrt(2)/2*U; V = sqrt(2)/2*V;


D1 = D1p; D2 = D2p;

for iter = 1:count    
   
    %% fix F and P, update W and Z
    % calculate Z
    P = full(P);
    Z =  P'*X./(sum(P',2)*ones(1,size(X,2))); %origin anchor \widetilde{Z}
    X_new_1 = [X;Z];
    A_p(1:n,n+1:end)=P;
    A_p(n+1:end,1:n)= P';
    D_p = diag(sum(A_p));
    L_p = D_p - A_p;
    P_1 = X_new_1' * L_p * X_new_1;
    % update W
    PP = P_1 - gamma * St;
    [W] = eig1(PP, dim, 0, 0);
    W = W * diag(1./sqrt(diag( W'* W))); % d*d'


    %% fix W and Z, update F and P
    % update P
    Z_sub = Z * W;
    X_sub = X * W;
    dist1 = L2_distance_1(X_sub',Z_sub'); %each cloumn is a data point.
    dist2 = L2_distance_1((D1 * U)',(D2p * V)'); 
    dist_all = dist1 + lambda * dist2;
    distance = dist_all;
    [a,~] = sort(distance,2);
    distance(distance>a(:,k)) = 0;
    distance(distance>0)=1;
    P = distance;
    % update F
    P = sparse(P);
    P = P + eps;
    p2 = sum(P,1);
    D2p = spdiags(1./sqrt(p2'),0,m,m);
    P1 = D1p*P*D2p;
    SS2 = P1'* P1;
    SS2 = full(SS2); 
    [V, ev0, ev]=eig1(SS2,m);
    V = V(:,1:c);
    U=(P1*V)./(ones(n,1)*sqrt(ev0(1:c)'));
    U = sqrt(2)/2*U; V = sqrt(2)/2*V;
    U_old = U; V_old = V;

    % % this way is to adjust alpha
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 < c - 0.0000001
        lambda = 2 * lambda;
    elseif fn2 > c+1-0.0000001
        lambda = lambda /2 ; U  = U_old ;V = V_old;
    else
        break
    end

end
U = U./(sum(U,2)*ones(1,c));
V = V./(sum(V,2)*ones(1,c));

P = sparse(P);