folder = 'C:\Users\lenovo\Desktop\ProjectImages';
filelist = dir(fullfile(folder, '*.png')); %get list of all jpg files in the folder
matnames = regexprep({filelist.name}, '\.jpg$', '.mat'); %replace .jpg by .mat in all file names

for n = 1:numel(filelist)
 ColorImage = imread(fullfile(folder, filelist(n).name)); %read images
 ColorImage = imresize(ColorImage, [25 25]);
 G = rgb2gray(ColorImage);
 I = ColorImage;
 Q(:,:,:,n) = I;
 Gr(:,:,n) = G; 
end

M1_color = reshape(Q, 625, 3, 723);
M1_gray = double(reshape(Gr, 625, 723))';
M1_red = double(reshape(Q(:,:,1,:), 625, 723))'; 
M1_green = double(reshape(Q(:,:,2,:), 625, 723))'; 
M1_blue = double(reshape(Q(:,:,3,:), 625, 723))'; 

M1_gray_train = M1_gray(1:700, :); 
M1_gray_test = M1_gray(701:723, :);
M1_red = M1_red(1:700, :);
M1_green = M1_green(1:700, :);
M1_blue = M1_blue(1:700, :);

redw = (M1_gray_train' * M1_gray_train)\( M1_gray_train' * M1_red); 
bluew = (M1_gray_train' * M1_gray_train)\( M1_gray_train' * M1_blue);
greenw = (M1_gray_train' * M1_gray_train)\( M1_gray_train' * M1_green);

rtest = reshape(M1_gray_test(16,:) * redw, 25, 25);
btest = reshape(M1_gray_test(16,:) * bluew, 25, 25);
gtest = reshape(M1_gray_test(16,:) * greenw, 25, 25);

rgbImage(:,:,1) = rtest; 
rgbImage(:,:,2) = gtest; 
rgbImage(:,:,3) = btest; 

imagesc(rgbImage/255)
