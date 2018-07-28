%% Read images from Path A and Path B
clear all;close all;clc;
a_H=[];
a_S=[];
a_V=[];
b_H=[];
b_S=[];
b_V=[];
for ii = 1:9
  close all;
  ii=10;
  pathology_a = imread(strcat('C:\Users\saurabh B\Documents\Proscia Challenge\pathology_A\'...
                 ,num2str(ii),'.png'));
               
  pathology_b = imread(strcat('C:\Users\saurabh B\Documents\Proscia Challenge\pathology_B\B'...
                 ,num2str(ii),'.png'));
               
               
 %% Convert to binary mask
 a_bw = imbinarize(rgb2gray(pathology_a));
 b_bw = imbinarize(rgb2gray(pathology_b));
 
 %% Complement
 a_bw_cmpl = imcomplement(a_bw);
 b_bw_cmpl = imcomplement(b_bw);
%  figure,imshow(a_bw_cmpl)
%  figure,imshow(b_bw_cmpl)
 
a_mask = repmat(a_bw_cmpl,[1 1 3]);
b_mask = repmat(b_bw_cmpl,[1 1 3]);

a_rgb = bsxfun(@times, pathology_a, cast(a_mask, 'like', pathology_a));
b_rgb = bsxfun(@times, pathology_b, cast(b_mask, 'like', pathology_b));

a_hsv = rgb2hsv(a_rgb);
b_hsv = rgb2hsv(b_rgb);

% % Pathology A
hue_a = a_hsv(:,:,1);
sat_a = a_hsv(:,:,2);
val_a = a_hsv(:,:,3);
non_zero_a = find(hue_a);

sumsin_a = sum(sind(hue_a(non_zero_a)));
sumcos_a = sum(cosd(hue_a(non_zero_a)));
averageH_a = atan2d(sumsin_a, sumcos_a); % Compute mean hue angle
averageH_a = averageH_a * 360;
a_H=[a_H,averageH_a];
a_S=[a_S,mean(sat_a(non_zero_a))];
a_V=[a_V,mean(val_a(non_zero_a))];
% % Pathology B
hue_b = b_hsv(:,:,1);
sat_b = b_hsv(:,:,2);
val_b = b_hsv(:,:,3);
non_zero_b = find(hue_b);
sumsin_b = sum(sind(hue_b(non_zero_b))); % Use (:) if you want to be able to handle 2-D hue images.
sumcos_b = sum(cosd(hue_b(non_zero_b)));
averageH_b = atan2d(sumsin_b, sumcos_b); 
averageH_b = averageH_b * 360;
b_H=[b_H,averageH_b];
b_S=[b_S,mean(sat_b(non_zero_b))];
b_V=[b_V,mean(val_b(non_zero_b))];
 
end