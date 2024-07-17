clc;
clear all;

buildingDir = fullfile('images');
buildingScene = imageDatastore(buildingDir);


montage(buildingScene.Files);

I = readimage(buildingScene , 1);
disp(buildingScene);

grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features , points] = extractFeatures(grayImage , points);

numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));


imageSize = zeros(numImages , 2);


for n=2:numImages

    pointsPrevious = points;
    featuresPrevious = features;

    I = readimage(buildingScene , n);
    
    grayImage = rgb2gray(I);

    imageSize(n,:) = size(grayImage);

    points = detectSURFFeatures(grayImage);
    [features , points] = extractFeatures(grayImage, points);

    indexPairs = matchFeatures(features, featuresPrevious , 'Unique',true);

    matchedPoints = points(indexPairs(:, 1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:, 2), :);

    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective' ,'Confidence', 99.9, 'MaxNumTrials', 2000);
    tforms(n).T = tforms(n).T * tforms(n-1).T;
end

disp(size(tforms));
disp('***');
disp(size(imageSize));

for i=1:numel(tforms) 
    [xlim(i, :), ylim(i, :)] = outputLimits(tforms(i), [1 imageSize(i, 2)], [1 imageSize(i, 1)]);
end


avgXLim = mean(xlim , 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end


for i = 1:numel(tforms)
    [xlim(i, :), ylim(i, :)] = outputLimits(tforms(i), [1 imageSize(i, 2)], [1 imageSize(i, 1)]);
end

maxImageSize = max(imageSize);

xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

width = round(xMax - xMin);
height = round(yMax - yMin);



panorama = zeros([height, width 3], 'like', I);

belender = vision.AlphaBlender('Operation', 'Binary Mask', 'MaskSource', 'Input port');

xLimits = [xMin, xMax];
yLimits = [yMin, yMax];
panoramaView = imref2d([height, width], xLimits, yLimits);


for i=1:numImages

    I = readimage(buildingScene, i);

    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    mask = imwarp(true(size(I, 1), size(I, 2)), tforms(i), 'OutputView', panoramaView);

    panorama = step(belender, panorama, warpedImage, mask);

end

figure
imshow(panorama);