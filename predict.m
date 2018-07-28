
 %% Convert to binary mask
 pathology_a = B1;
 a_bw = imbinarize(rgb2gray(pathology_a));

 
 %% Complement
 a_bw_cmpl = imcomplement(a_bw);
 
 
a_mask = repmat(a_bw_cmpl,[1 1 3]);

a_rgb = bsxfun(@times, pathology_a, cast(a_mask, 'like', pathology_a));


a_hsv = rgb2hsv(a_rgb);

hue_a = a_hsv(:,:,1);
sat_a = a_hsv(:,:,2);
val_a = a_hsv(:,:,3);
non_zero_a = find(hue_a);


sumsin_a = sum(sind(hue_a(non_zero_a)));
sumcos_a = sum(cosd(hue_a(non_zero_a)));
averageH_a = atan2d(sumsin_a, sumcos_a); % Compute mean hue angle
averageH_a = averageH_a * 360;
a_H = averageH_a;
a_S = mean(sat_a(non_zero_a));
a_V = mean(val_a(non_zero_a));

if a_H > 300
  disp('A')
else
  disp('B')
end



