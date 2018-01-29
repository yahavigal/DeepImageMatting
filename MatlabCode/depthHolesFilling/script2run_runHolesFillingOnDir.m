% script to run runHolesFillingOnDir
% path 2 the main directory where clor,gt and depth are
path2data = '/media/or/1TB-data/cc_067_no_shifts/DataSet_2_composed';
mainDirName = 'DataSet_2_composed';
% path to depth after holes filling
d_outName = 'depth_filled';
% path to normalized low resolution depth after holes filling
dn_outName = 'depth_norm_v3';

runHolesFillingOnDir ( path2data, mainDirName, d_outName, dn_outName);