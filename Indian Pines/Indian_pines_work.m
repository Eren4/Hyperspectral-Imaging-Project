%%
clc; clear; close all;

%%
load Indian_pines.mat; load Indian_pines_gt.mat;

%% Step 2 - Normalizing the spectral signals

% Create temporary matrix
spectral_signals = zeros(21025, 220);

% Allocate the spectral signals to the temporary matrix
k = 1;
for i = 1 : 145
    for j = 1 : 145
        spectral_signals(k, :) = indian_pines(i, j, :);
        k = k + 1;
    end
end

% Normalize the signals that are stored in the temporary matrix
for i = 1 : 21025
    spectral_signals(i, :) = spectral_signals(i, :) / norm(spectral_signals(i, :));
end

% Assign the normalized values to the hyperspectral cube
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines(i, j, :) = spectral_signals(k, :);
        k = k + 1;
    end
end

%% Step 3 - Select 3 suitable bands at different wavelengths from the dataset, create an appropriate false-color image and show it.
band50 = indian_pines(:, :, 50);
band27 = indian_pines(:, :, 27);
band17 = indian_pines(:, :, 17);
bands = zeros(145, 145, 3);
bands(:, :, 1) = band50;
bands(:, :, 2) = band27;
bands(:, :, 3) = band17;
figure; imshow(bands);

%% Step 4 - Visualize the ground truth data. Assign a different color to each label to distinguish the classes properly in the image.
figure; imagesc(indian_pines_gt);
colormap([0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 0.5 0; 0.5 1 0; 0 1 0.75; 0.5 0 1; 1 0 0.5; 0.5 0.5 0.5; 0.2 0.9 0.4; 0.5 0.2 0.7; 0.1 0.2 1]);

%% Step 5 - Present the number of elements belonging to each class using histogram plot. Exclude the 0 valued entities
index = indian_pines_gt > 0;
figure; histogram(indian_pines_gt(index));

%% Step 6 - Apply PCA to the HSI dataset and reduce the band number to 3, 5, 10, 20 and 25, respectively.
[coeff, score, latent] = pca(spectral_signals);
% 3 bands
spectral_signals3 = score(:, 1:3);
indian_pines3 = zeros(145, 145, 3);
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines3(i, j, :) = spectral_signals3(k, :);
        k = k + 1;
    end
end
% 5 bands
spectral_signals5 = score(:, 1:5);
indian_pines5 = zeros(145, 145, 5);
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines5(i, j, :) = spectral_signals5(k, :);
        k = k + 1;
    end
end
% 10 bands
spectral_signals10 = score(:, 1:10);
indian_pines10 = zeros(145, 145, 10);
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines10(i, j, :) = spectral_signals10(k, :);
        k = k + 1;
    end
end
% 20 bands
spectral_signals20 = score(:, 1:20);
indian_pines20 = zeros(145, 145, 20);
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines20(i, j, :) = spectral_signals20(k, :);
        k = k + 1;
    end
end
% 25 bands
spectral_signals25 = score(:, 1:25);
indian_pines25 = zeros(145, 145, 25);
k = 1;
for i = 1 : 145
    for j = 1 : 145
        indian_pines25(i, j, :) = spectral_signals25(k, :);
        k = k + 1;
    end
end

%% Step 7 - Generate another false color image for the dataset by using the 3-band version obtained using PCA.
figure; imshow(indian_pines3);

%% Step 8 - Create similarity matrices and heat maps for each reduced data obtained in step 6 and determine which class is closer to which class.
% 3 band
Y3 = pdist(spectral_signals3');
Y3_square = squareform(Y3);
figure; heatmap(Y3_square);
% Classes 2 and 3 are the most similar
% 5 band
Y5 = pdist(spectral_signals5');
Y5_square = squareform(Y5);
figure; heatmap(Y5_square);
% Classes 4 and 5 are the most similar
% 10 band
Y10 = pdist(spectral_signals10');
Y10_square = squareform(Y10);
figure; heatmap(Y10_square);
% Classes 9 and 10 are the most similar
% 20 band
Y20 = pdist(spectral_signals20');
Y20_square = squareform(Y20);
figure; heatmap(Y20_square);
% Classes 19 and 20 are the most similar
% 25 band
Y25 = pdist(spectral_signals25');
Y25_square = squareform(Y25);
figure; heatmap(Y25_square);
% Classes 24 and 25 are the most similar

%% Step 9 - SVM