function [ P ] = PF( W,a,apr_lvl,parallel )
%PF gives the P-values prescribed by the Polya Filter for each link of the network 
%   W: is the adjecency matrix 
%   a: is the free parameter of the filter
%      note1: if a<0 estimates the free parameter using Maximum Likelihood extrimation
%      note2: when a=1 the Polya Filter concides with the disparity filter
%   apr_lvl: is a constant defining the regime under which the approximate form of the p-value (Eq. (6) of the paper below) can be used.
%     For example, setting apr_lvl = 10 will trigger the use of the approximate form for every link such that s > 10*k/a, w > 10, and s-w>10*k/a. 
%     The approximate form of the p-value is much faster to compute. If not given it is 
%     automatically set to apr_lvl=10, if apr_lvl = 0 the approximate form is always used.
%     The option apr_lvl=0 is prescribed (and automatically done) in networks with non-intger
%     weights.
%   parallel: 0 => no parallel computing, ~=0 => parallel computing    

% The Polya Filter is described in the academic paper:
% "A Pólya urn approach to information filtering in complex networks" by R. Marcaccioli and G. Livan available at https://rdcu.be/bmIjz

% NOTE: The precision of the calculation is set to the precision of the machine (usually 1e-12).
%  to reach the precision reached in the paper we used the Matlab toolbox "Multiple Precision Toolbox"

%check if the network is symmetric (i.e. undirected) and get the edge list for both cases
if issymmetric(W)==1
    U = triu(W);
    [i,j,w] = find(U); 
else
    [i,j,w] = find(W); 
end

%get the degrees and strengths
k_in(:,1) = full(sum(W~=0,1));
s_in(:,1) = full(sum(W,1));
k_out(:,1) = full(sum(W~=0,2));
s_out(:,1) = full(sum(W,2));

k_in = k_in(j);
k_out = k_out(i);
s_in = s_in(j);
s_out = s_out(i);

%if a<0 get the ML estimates
if a<0
   disp('Starting the ML estimation')
   [a,err] = get_ML_estimate(W);
   disp(['Estimation ended, a = ', num2str(a)])
end

%use the asymptotic form if non integer wights are present
if sum(mod(w,1))~=0
    apr_lvl = 0;
end

%num_edges = size(W)

%calculate p-values
p(:,1) = polya_cdf(w,s_in,k_in,a,apr_lvl,parallel);
p(:,2) = polya_cdf(w,s_out,k_out,a,apr_lvl,parallel);

P = min(p(:,1),p(:,2));

%handle the case k=1
P(k_in==1) = p(k_in==1,2);
P(k_out==1) = p(k_out==1,1);
P((double(k_in==1)+double(k_out==1))==2) = 1;

end

function [p] = polya_cdf(w,s,k,a,L,parallel)

p = nan(length(w),1);

if a==0 %binomial case
    p = binocdf(w,s,1./k);
else
    %check where approximation can be performed
    idx1 = s-w>=L*((k-1)./a+1);
    idx2 = w>=L*max(1./a,1);
    idx3 = s>=L*max(k./a,1);
    idx4 = k>=L*(a-1+1e-20);
    idx = (double(idx1)+double(idx2)+double(idx3)+double(idx4))==4;

    %calculate the p-values that can be approximated
    p(idx) = 1/gamma(1/a)*(1-w(idx)./s(idx)).^((k(idx)-1)/a).*(w(idx).*k(idx)./(s(idx)*a)).^(1/a-1);

    %calculate the p-values that cannot be aprroximated
    idx = find(double(idx)==0);
    if isempty(idx)==0
        if parallel==0
            for ii=1:length(idx)
                n = s(idx(ii));
                A = 1/a;
                B = (k(idx(ii))-1)./a;
                x = (0:1:w(idx(ii))-1)';
                p(idx(ii)) = 1 - sum(exp(gammaln(n+1)+betaln(x+A,n-x+B)-gammaln(x+1)-gammaln(n-x+1)-betaln(A,B)));
            end
        else
            aux = nan(length(idx),1);
            parfor ii=1:length(idx)
                n = s(idx(ii));
                A = 1/a;
                B = (k(idx(ii))-1)./a;
                x = (0:1:w(idx(ii))-1)';
                aux(ii,1) = 1 - sum(exp(gammaln(n+1)+betaln(x+A,n-x+B)-gammaln(x+1)-gammaln(n-x+1)-betaln(A,B)));
            end
            p(idx) = aux;
        end
    end
end

%catch rounding errors (use the mp-toolbox for higher prcision)
p(p<0) = 0;

end

function [a_best,err] = get_ML_estimate(W) 
%This function can be used to get the maximum likelihood estimation of the
%free parameter "a". It will set the filter on the network's own
%heterogeneity and it will therefore produce ultra-sparse backbones
%   W: is the adjecency matrix

%check if the network is symmetric (i.e. undirected) and get the edge list for both cases
if issymmetric(W)==1
    U = triu(W);
    [i,j,w] = find(U); 
else
    [i,j,w] = find(W); 
end

%get the degrees and strengths
k_in(:,1) = full(sum(W~=0,1));
s_in(:,1) = full(sum(W,1));
k_out(:,1) = full(sum(W~=0,2));
s_out(:,1) = full(sum(W,2));

w = [w;w];
k = [k_out(i);k_in(j)];
s = [s_out(i);s_in(j)];

%get rid of the links with degree 1
w(k==1) = [];
s(k==1) = [];
k(k==1) = [];

f = @(a)eq_from_lhood([w k s],a);
g = @(a)lhood([w k s],a);
options_lsq = optimoptions(@lsqnonlin,'Display','off');

%first find the maximum value of the likelihood
x0 = lsqnonlin(g,0.5,0,15,options_lsq);
%use this value to calculate the value that put the derivative to 0
[a_best,err] = lsqnonlin(f,x0,0,15,options_lsq);

%try higher precision if feval is not close to 0
if err>1e-6
    options_lsq = optimoptions(@lsqnonlin,'MaxFunctionEvaluations',2000,'FunctionTolerance',1e-15,'StepTolerance',1e-25,'OptimalityTolerance',1e-25,'Display','off');
    [a_best,err] = lsqnonlin(f,x0,0,15,options_lsq);
    if err>1e-6
        disp('Try stricter minimization options')
    end
end

end

function [ out ] = eq_from_lhood( X,a )
%derivative of the likelihood that must be put equal to 0
    w = X(:,1);
    k = X(:,2);
    s = X(:,3);
    DL = a.^(-2).*(psi(a.^(-1))+((-1)+k).*psi(a.^(-1).*((-1)+k))+(-1).*k.*psi(a.^(-1).*k)+k.*psi(a.^(-1).*k+s)+(-1).*psi(a.^(-1)+w)+(-1).*((-1)+k).*psi(a.^(-1).*((-1)+k+a.*s+(-1).*a.*w)));
    DL(isnan(DL)) = 0;
    out = double(sum(DL));
    if isinf(out)==1
        out=1e100;
    end
end

function [ out ] = lhood( X,a )
%likelihood that needs to be maximised
    w = X(:,1);
    k = X(:,2);
    s = X(:,3);
    P = polya_pdf(w,s,k,a);
    P(P==0)=1e-20;
    out = sum(-log(P));
    if isinf(out)==1
        out = sign(out)*1e100;
    end
end

function [p] = polya_pdf(w,s,k,a)
%pdf of the distribution
if a==0 %binomial case
    p = binocdf(w,s,1./k);
else       
    n = s;
    A = 1/a;
    B = (k-1)./a;
    x = w;
    p = exp(gammaln(n+1)+betaln(x+A,n-x+B)-gammaln(x+1)-gammaln(n-x+1)-betaln(A,B));
end
end