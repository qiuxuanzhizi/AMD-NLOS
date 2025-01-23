function ximage = AMD_solution(mask, y, psf, jit, samp, tau, eta, miniter, maxiter)

[bin, N, N] = size(y);
bin_resolution = 32e-12;            % Time resolution
wall_size = 1; % optional
width = wall_size/2;
c = 3*10^8;        % speed of light
range = bin.*c.*bin_resolution;  %
slope = width./range;
[mtx,mtxi] = resamplingOperator(bin);
mtx = full(mtx);
mtxi = full(mtxi);

mask = definemask(N,samp);
mask = reshape(mask,[1,N,N]);
mask = repmat(mask,[bin 1 1]);
mask = single(mask);
%disp(size(mask))

% utility functions
S = 1;
trim_array = @(x) x(S*bin/2+1:end-S*bin/2, N/2+1:end-N/2, N/2+1:end-N/2);
pad_array = @(x) padarray(x, [S*bin/2, N/2, N/2]);
square2cube = @(x) reshape(x, [],N,N);
cube2square = @(x) x(:,:);
vec = @(x) x(:);
Ffun = @(x)  fftn(x);
Ftfun = @(x) ifftn(x);

pad_array1 = @(x) padarray(x, [bin/2, N/2-1, N/2-1],'pre');
pad_array2 = @(x) padarray(x, [bin/2, N/2+1, N/2+1],'post');
trim_array1 = @(x) x(bin/2+1:3*bin/2, N/2:3*N/2-1, N/2:3*N/2-1);
trim_arrayjit = @(x) x(4:3+bin,:,:);
psf = permute(psf,[3 2 1]);
%disp(size(psf))
psf = single(psf);
psf1 = padarray(psf,[0,1,1],'pre');
psfFT = fftn(psf1);
psfFT1 = fftn(padarray(flip(flip(flip(psf,1),2),3),[0,1,1],'pre'));
jit = permute(jit,[3 2 1]);
jitFT = fftn(padarray(jit,[bin-1,N-1,N-1],'post'));
    
A = @(x) real(trim_arrayjit(Ftfun(jitFT.*Ffun(padarray(square2cube(mtxi*cube2square(trim_array1(real(ifftshift(Ftfun(psfFT .* Ffun(pad_array2(pad_array1(x))))))))),[6 0 0],'post'))))).*mask;  
AT = @(x) trim_array1(real(ifftshift(Ftfun(psfFT1 .* Ffun(pad_array2(pad_array1(square2cube(mtx*cube2square(x)))))))));

%% Initialization
u = zeros(size(y));
alpha = [1/2 1/2];
sigma2 = [0.005 10];


omega2 = zeros(size(y));
omega2(y>=.95) = 1;
omega2(y<=0.05) = 1;
omega1 = 1-omega2;
omega(:,:,:,1) = omega1;
omega(:,:,:,2) = omega2;

w = omega(:,:,:,1)/(sigma2(1)+eps) + omega(:,:,:,2)/(sigma2(2)+eps);

iter = 1;
mu = 0;
subtolerance = 1e-5;
%% Begin Main Algorithm Loop
while (iter <= miniter) || (iter <= maxiter)
    disp(iter)
    %sub-problem 1
    f = subsolution1(u, tau, eta, mu, subtolerance);
    f(f<0) = 0;
    f(f>1) = 1;

    %sub-problem 2
    b = AT(w.*y) + eta*f;
    u = subsolution2(A,AT,eta,b,w,u);
    u(u<0)=0;
    u(u>1)=1;

    %sub-problem 3
    d = A(u) - y;
    [alpha, sigma2, omega] = subsolution3(d, alpha, sigma2);
    w=omega(:,:,:,1)/(sigma2(1)+eps)+omega(:,:,:,2)/(sigma2(2)+eps);

    iter = iter + 1;
    vol(:,:,:) = abs(f(:,:,:));
    vol(end-50:end, :, :) = 0;
    figure(1);draw3D(vol,0.5,0.2,1);drawnow
end
ximage = f;

function [mtx,mtxi] = resamplingOperator(M)   % refer to lct reconstruction
 % Local function that defines resampling operators
     mtx = sparse([],[],[],M.^2,M,M.^2);
     
     x = 1:M.^2;
     mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
     mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
     mtxi = mtx';
     
     K = log(M)./log(2);
     for k = 1:round(K)
          mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
          mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
     end


 function mask = definemask(N,samp)
    xx = linspace(1,N,samp);
    yy = linspace(1,N,samp);
    mask = zeros(N,N); 
    for i = 1:samp
        for j = 1:samp
            mask(round(xx),round(yy)) = 1;
        end
    end

