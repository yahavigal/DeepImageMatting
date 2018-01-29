clear
clc
close all

path2data = '\\ger\ec\proj\ha\perc\BGS_data\Alexandra\forDepthFilling\set1\main\w11525774\sitting';

isSave = 1; 
subdirName = 'depth_norm_v3';
dir2save = replace(path2data,'main',subdirName);

if isSave
    mkdir( dir2save);
end

d = dir( fullfile( path2data, '*depth.png' ));
numImgs = numel(d);
e = 0;
for j1 = 66 : numImgs
    [j1 numImgs]
    depth = imread( fullfile( path2data, d(j1).name) );
    t = cputime;
    [ d_n_lr, d_f ] = fillHolesAndNormalizeDepth( depth);
    e = e + cputime-t ;

    
    imF = imfuse(depth, d_f, 'montage', 'Scaling','joint' );
    [h,w] = size(imF);
    h  = figure('Name', d(j1).name, 'Position', [270 340 w h ]); imagesc(imF);
    waitfor(h);
    close all;
    
    if isSave == 1
        path2save = fullfile( dir2save, d(j1).name);
        imwrite( d_f, path2save);
        path2save = fullfile( dir2save,['lr_' d(j1).name]);
        imwrite( d_n_lr, path2save);
    end
    
end
e/numImgs