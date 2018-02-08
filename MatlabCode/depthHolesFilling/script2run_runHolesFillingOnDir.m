% script to run runHolesFillingOnDir
% path 2 the main directory where clor,gt and depth are
path2data = '/media/or/1TB-data/cc_067_no_shifts/DataSet_2_composed';
mainDirName = 'DataSet_2_composed'; % 'Set1_07_2017';
% path to depth after holes filling
d_outName = 'depth_filled_v2'; %'Set1_07_2017_depth_filled_v2';
% path to normalized low resolution depth after holes filling
dn_lr_outName = ''; % 'Set1_07_2017_depth_norm_v3_lr_v2';
dn_fr_outName = '';% 'Set1_07_2017_depth_norm_v3_v2';

et = runHolesFillingOnDir ( path2data, mainDirName, d_outName, dn_lr_outName, dn_fr_outName);