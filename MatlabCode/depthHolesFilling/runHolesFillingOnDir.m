function [et] = runHolesFillingOnDir ( path2data, mainDirName, d_outName, dn_lr_outName, dn_fr_outName)
% run over all subdirectories in dir

d = dir( fullfile( path2data,'*depth.png' ));
numImgs = numel(d);
et = -1;
if numImgs == 0
    d = dir(  path2data );
    for i1 = 3 : numel(d)
        if d(i1).isdir
            path2data_curr = fullfile( path2data,d(i1).name);
            et = runHolesFillingOnDir ( path2data_curr, mainDirName, d_outName, dn_lr_outName, dn_fr_outName);
        end
    end
else
	if ~isempty(d_outName)
        dir2save_d = replace(path2data,mainDirName,d_outName);
        mkdir( dir2save_d);
    end
    if ~isempty(dn_lr_outName)
        dir2save_dn_lr = replace(path2data,mainDirName,dn_lr_outName);
        mkdir( dir2save_dn_lr);
    end
    if ~isempty(dn_fr_outName)
        dir2save_dn_fr = replace(path2data,mainDirName,dn_fr_outName);
        mkdir( dir2save_dn_fr);
    end
    % loop over all depth images in directory
    et = 0;
    for j1 = 1 : numImgs
        % [j1 numImgs]
        path2depth = fullfile( path2data, d(j1).name);
        depth = imread( path2depth );
        t = cputime;
        [ d_n_lr, d_f, d_n_fr ] = fillHolesAndNormalizeDepth( depth);
        et = et + cputime-t ;
        et/j1
        
        if ~isempty(d_outName)
            path2save = fullfile( dir2save_d, d(j1).name);
            tf = strcmp(path2depth,path2save);
            if tf == true
                error('pathes are the same');
            end
            imwrite( d_f, path2save);
        end
        
        if ~isempty(dn_lr_outName)
            path2save = fullfile( dir2save_dn_lr,d(j1).name);
            tf = strcmp(path2depth,path2save);
            if tf == true
                error('pathes are the same');
            end
            imwrite( d_n_lr, path2save);
        end
        
        if ~isempty(dn_fr_outName)
            path2save = fullfile( dir2save_dn_fr,d(j1).name);
            tf = strcmp(path2depth,path2save);
            if tf == true
                error('pathes are the same');
            end
            imwrite( d_n_fr, path2save);
        end
    end
    et = et/numImgs
end
