clear
clc
close all

path2data = '/media/or/1TB-data/deepImageMatting/Set1_07_2017/w11513762/walking/';

isSave = 0; 
mainDirName = 'Set1_07_2017';
subdirName = 'depth_norm_v3_2';
dir2save = replace(path2data,mainDirName,subdirName);

if isSave
    mkdir( dir2save);
end

d = dir( fullfile( path2data, '*depth.png' ));
numImgs = numel(d);
e = 0;
for j1 = 4 : numImgs
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