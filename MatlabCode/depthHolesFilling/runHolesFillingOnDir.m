function [] = runHolesFillingOnDir ( path2data, mainDirName, d_outName, dn_lr_outName, dn_fr_outName)
% run over all subdirectories in dir

d = dir( fullfile( path2data,'*depth.png' ));
numImgs = numel(d);
if numImgs == 0
    d = dir(  path2data );
    for i1 = 3 : numel(d)
        if d(i1).isdir
            path2data_curr = fullfile( path2data,d(i1).name);
            runHolesFillingOnDir ( path2data_curr, mainDirName, d_outName, dn_lr_outName, dn_fr_outName);
        end
    end
else
    dir2save_d = replace(path2data,mainDirName,d_outName);
    dir2save_dn_lr = replace(path2data,mainDirName,dn_lr_outName);
    dir2save_dn_fr = replace(path2data,mainDirName,dn_fr_outName);
    
    mkdir( dir2save_d);
    mkdir( dir2save_dn_lr);
    mkdir( dir2save_dn_fr);
    % loop over all depth images in directory
    for j1 = 1 : numImgs
        [j1 numImgs]
        depth = imread( fullfile( path2data, d(j1).name) );
        [ d_n_lr, d_f, d_n_fr ] = fillHolesAndNormalizeDepth( depth);
        
        path2save = fullfile( dir2save_d, d(j1).name);
        imwrite( d_f, path2save);
        
        path2save = fullfile( dir2save_dn_lr,d(j1).name);
        imwrite( d_n_lr, path2save);
        
        path2save = fullfile( dir2save_dn_fr,d(j1).name);
        imwrite( d_n_fr, path2save);
    end
    
end
end