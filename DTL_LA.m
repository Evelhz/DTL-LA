function [Xs, Xt, A, Att] = DTL_LA(Xs, Xt, Ys, Yt0, Yt, options)

alpha = options.alpha;
mu = options.mu;
beta = options.beta;
gamma = options.gamma;
eta = options.eta;
ker = 'primal';
k = options.k;
T = options.T;
K1 = options.K1;
K2 = options.K2;

m = size(Xs,1);
ns = size(Xs,2);
nt = size(Xt,2);

class = unique(Ys);
C = length(class);
if strcmp(ker,'primal')

    %--------------------------------------------------------------------------
    % compute LDA
    dim = size(Xs,1);
    meanTotal = mean(Xs,2);

    Sw = zeros(dim, dim);
    Sb = zeros(dim, dim);
    for i=1:C
        Xi = Xs(:,find(Ys==class(i)));
        meanClass = mean(Xi,2);
        Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
        Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
        Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
    end
    P = zeros(2*m,2*m);
    P(1:m,1:m) = Sb;
    Q = Sw;

    %----------------------Begining of Locality Alignment----------------------
    if options.eta > 0
        Xs_tmp = Xs';
        if size(Xs_tmp,2)==2048
            reduced_dim = min(size(Xs_tmp,1),301)-1;
            coef = pca(Xs_tmp);
            Xs_tmp = Xs_tmp*coef(:,1:reduced_dim);
        end
        if size(Ys,1) == 1
            Ys = Ys';
        end
        [nTrain,nDim]=size(Xs_tmp);
        patch=zeros(nTrain, K1+K2+1);
        for n=1:nTrain
            temp = Xs_tmp(n,:);
            c1=find(Ys==Ys(n));
            c2=find(Ys~=Ys(n));
            dist=pdist2(temp, Xs_tmp(c1,:));%computer distance between one sample and the others
            [sorted,index] = sort(dist');
            cindex=[c1(index(2:K1+1))];
            dist=pdist2(temp, Xs_tmp(c2,:));%computer distance between one sample and the others
            [sorted,index] = sort(dist');
            %     dindex=[c2(1:k2)];
            dindex=[c2(index(1:K2))];
            patch(n,:)=[n cindex' dindex']; %build patch with index of the measurement,k1 neighbors belonging to the same class, k2 belonging to the different class
        end
        % construct the alignment matrix L
        beta2=0.05;
        delta=0.01;
        w=[ones(1,K1) -beta2*ones(1,K2)]';
        Li=[sum(w),-w';-w,diag(w)];
        t=10;
        mi=exp(-1/((K2+delta)*t));
        % L=zeros(nTrain,nTrain);%initialize L=0;
        L=zeros(nTrain,nTrain);
        for i=1:nTrain
            %compute sample weighting
            L(patch(i,:),patch(i,:))= L(patch(i,:),patch(i,:))+mi*Li;
        end
        L = Xs*L*Xs';
    else
        L = 0;
    end
    %-----------------------End of Locality Alignment--------------------------

    for t = 1:T
        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

        Ts = Xs*Ms*Xs';
        Tt = Xt*Mt*Xt';
        Tst = Xs*Mst*Xt';
        Tts = Xt*Mts*Xs';

        % Construct centering matrix
        Ht = eye(nt)-1/(nt)*ones(nt,nt);

        X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];
        H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];

        Smax = mu*X*H*X'+beta*P;
        Smin = [Ts+alpha*eye(m)+beta*Q+eta*L, Tst-alpha*eye(m) ; ...
            Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');
        A = W(1:m, :);
        Att = W(m+1:end, :);

        Zs = A'*Xs;
        Zt = Att'*Xt;

        if T>1
            if isempty('Yt')
                continue;
            end
            mdl = fitcknn(Zs',Ys);
            [Yt0,score,cost] = predict(mdl,Zt');
            acc = length(find(Yt0==Yt))/length(Yt);
            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
        end
    end
else

    Xst = [Xs, Xt];
    nst = size(Xst,2);
    [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);
    %--------------------------------------------------------------------------
    % compute LDA
    dim = size(Ks,2);
    C = length(class);
    meanTotal = mean(Ks,1);

    Sw = zeros(dim, dim);
    Sb = zeros(dim, dim);
    for i=1:C
        Xi = Ks(find(Ys==class(i)),:);
        meanClass = mean(Xi,1);
        Hi = eye(size(Xi,1))-1/(size(Xi,1))*ones(size(Xi,1),size(Xi,1));
        Sw = Sw + Xi'*Hi*Xi; % calculate within-class scatter
        Sb = Sb + size(Xi,1)*(meanClass-meanTotal)'*(meanClass-meanTotal); % calculate between-class scatter
    end
    P = zeros(2*nst,2*nst);
    P(1:nst,1:nst) = Sb;
    Q = Sw;

    for t = 1:T

        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

        Ts = Ks'*Ms*Ks;
        Tt = Kt'*Mt*Kt;
        Tst = Ks'*Mst*Kt;
        Tts = Kt'*Mts*Ks;

        K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
        Smax =  mu*K'*K+beta*P;

        Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
            Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), k, 'LM');
        W = real(W);

        A = W(1:nst, :);
        Att = W(nst+1:end, :);

        Zs = A'*Ks';
        Zt = Att'*Kt';

        if T>1
            mdl = fitcknn(Zs',Ys);
            [Yt0,score,cost] = predict(mdl,Zt');
            acc = length(find(Yt0==Yt))/length(Yt);
            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
        end
    end
end
Xs = Zs;
Xt = Zt;

