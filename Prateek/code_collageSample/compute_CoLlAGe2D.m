
function [collage_map, volfeats]=compute_CoLlAGe2D(origImage, tumorMask, winRadius,haralick_number)

%(c) Prateek Prasanna, 2016
% Related publication: Prasanna, Prateek, et. al.,
% "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): a new radiomics descriptor." 
% Scientific reports 6 (2016): 37241.

I=origImage;
mask=tumorMask;

%imshow(mask)
[r_mask,c_mask]=find(mask==1);
c_min=min(c_mask);c_max=max(c_mask);r_min=min(r_mask);r_max=max(r_mask);

x1=r_min;
x2=r_max;
y1=c_min;
y2=c_max;

if (x1-winRadius<1) || (y1-winRadius<1) || (x2+winRadius>size(I,1)) || (y2+winRadius>size(I,2))
    warning('COLLAGE: Gradient would go outside image border. Cannot compute CoLlAGe - Please erode mask. Returning NANs');
    collage_map = nan([size(mask),max(haralick_number)]);
    volfeats = [];
    return
end

    I2_outer = I(max(x1-winRadius,1):min(x2+winRadius,size(I,1)),max(y1-winRadius,1):min(y2+winRadius,size(I,2)));
    I2_double_outer=im2double(I2_outer);
    I2_inner=I(x1:x2,y1:y2);
    [r, c]=size(I2_inner);
    [Gx, Gy]=gradient(I2_double_outer);

[dominant_orientation_roi]=find_orientation_CoLlAGe_2D(Gx,Gy,winRadius,r,c);
BW1=mask(x1:x2,y1:y2);
BW1=double(BW1);


% Find co-occurrence features of orientation
%clear volfeats;
haralickfun=@haralick2mex;
vol=double(dominant_orientation_roi);

  
nharalicks=13;  % Number of Features
bg=-1;   % Background-1
ws=2*winRadius+1;    % Window Size
hardist=1;   % Distance to search in a window
harN=64;     % Maximum number of quantization level 64
volN=round(rescale_range(vol,0,harN-1));   % Quantizing an image
% volN(~volN) = 1; 
% volN
addedfeats=0;  % Feature counter index

volfeats = zeros(size(volN, 1), size(volN, 2), 13);
%% 
volfeats(:,:,addedfeats+(1:nharalicks))=haralickfun(volN,harN,ws,hardist,bg);
collage_map = nan([size(mask),max(haralick_number)]);
collage_map(x1:x2,y1:y2,haralick_number) = volfeats(:,:,haralick_number);





