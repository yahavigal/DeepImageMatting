function [depth_out] = interpolateLineAtEdges( depth_in)
[ h, w] = size(depth_in);
% all lines at edges
depth_out = horLine(depth_in, 1);
depth_out = horLine(depth_out, h);
depth_out = vertLine(depth_out, 1);
depth_out = vertLine(depth_out, w);

end