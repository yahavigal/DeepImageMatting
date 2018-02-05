% script to run runHolesFillingOnDir
% path 2 the main directory where clor,gt and depth are
path2data = '/media/or/Data/deepImageMatting/Set1_07_2017';
mainDirName = 'Set1_07_2017';
% path to depth after holes filling
d_outName = 'Set1_07_2017_depth_filled';
% path to normalized low resolution depth after holes filling
dn_lr_outName = 'Set1_07_2017_depth_norm_v3_lr';
dn_fr_outName = 'Set1_07_2017_depth_norm_v3';

et = runHolesFillingOnDir ( path2data, mainDirName, d_outName, dn_lr_outName, dn_fr_outName);