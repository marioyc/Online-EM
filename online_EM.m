function [mu, Sigma, weight] = online_EM(X, k)
[n, d] = size(X); 

% initialization
mu = rand(k, d);
Sigma = zeros(d, d, k);
for j = 1:k
    Sigma(:,:,j) = eye(d,d);
end
weight = ones(1,k)/k;
w_bar = zeros(1,k);
w_bar_x = mu;
w_bar_x2 = Sigma;


for i = 1:n
    lr = 1/i;
    % E step
    for j = 1:k
        w_bar(j) = weight(j)*mvnpdf(X(i,:), mu(j, :), Sigma(:,:,j));
    end
    w_bar = w_bar/sum(w_bar);
    
    % M step
    weight = weight + lr*(w_bar-weight);
    w_bar_x = w_bar_x + lr*(w_bar'*X(i,:) - w_bar_x);
    for j = 1:k
        w_bar_x2(:,:,j) = w_bar_x2(:,:,j) + lr*(w_bar(j)*X(i,:)'*X(i,:)-w_bar_x2(:,:,j));
    end
    
    if i > 20
%         mu = w_bar_x./repmat(w_bar', [1,d]);
        for j = 1:k
            mu(j,:) = w_bar_x(j,:)/weight(j);
            Sigma(:,:,j) = (w_bar_x2(:,:,j) - w_bar_x(j,:)'*mu(j,:))/weight(j);
            Sigma(:,:,j) = (Sigma(:,:,j)+Sigma(:,:,j)')/2;
        end
    end
    
end


end