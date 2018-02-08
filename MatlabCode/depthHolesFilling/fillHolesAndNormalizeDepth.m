function [depth_out_lr, depth_out, depth_norm] = fillHolesAndNormalizeDepth( depth)
% fill holes and depth and normalize
%   Detailed explanation goes here
resizeFactor = 0.25;
depth2proc = imresize(depth, resizeFactor, 'nearest');
[ h, w] = size(depth2proc);
d_m = medfilt2(depth2proc);
d_tmp = d_m;
d_tmp = interpolateLineAtEdges( d_tmp);
if 1
    d_tmp(d_tmp == 0) = 10000;
    minVal = min(d_tmp(:));
    d_tmp (d_tmp > (minVal + 900) ) = 0;
end
% d_tmp = medfilt2(d_tmp);

[B,L,N,A] = bwboundaries(d_tmp);
% go only over wholes
% figure; imagesc(d_tmp)
for i1 = N +1 : length(B)
    cc = B{i1};
        
    X = cc(:,2);
    Y = cc(:,1);
    minX = min(X); minY = min(Y); maxX = max(X); maxY = max(Y);
    ws = max( 1, (( maxX - minX) + (maxY - minY)) /6 ) ; %  max(2, (( maxX - minX) + (maxY - minY)) /4 ) ;    
    ws = 2*round(ws) + 1; % round(ws) + 1; %
    
    [ numPts, ~] = size(cc);
%     if numPts > 400
%         continue;
%     end
 
    ws = min(11, ws);
    ws0 = floor(ws/2);
    
    indYstart = max(1, minY - ws0);
    ws_y_s = minY - indYstart; 
    indYend = min(h,  maxY + ws0);
    ws_y_e = indYend - maxY;
    indXstart = max(1,minX - ws0);
    ws_x_s = minX - indXstart; 
    indXend = min(w,maxX + ws0);
    ws_x_e = indXend - maxX;
    
%     if ws_y_s ~= ws | ws_y_e ~= ws | ws_x_s ~= ws | ws_x_e ~= ws
%         ws = ws;
%     end
    croi = d_tmp( indYstart :indYend, indXstart : indXend);
    
    if numPts > 60
        % test if it fillable
        
        vt = croi(:);
        vt(vt==0) = [];
        averVal = mean(vt);
        % stdVal = std(double(vt));
if 1        
        tmp = croi;
        tmp = medfilt2(tmp, [5,5]);
end

if 0 
       tmp = d_m(indYstart :indYend, indXstart : indXend);
end
        
if 1
        cc_sub = [];
        tmp1 = d_m(indYstart :indYend, indXstart : indXend);        
        cc_sub(:,1) = cc(:,1) - indYstart;
        cc_sub(:,2) = cc(:,2) - indXstart;
        [ m, n] = size(tmp);
        inside_mask = poly2mask(cc_sub(:,2),cc_sub(:,1),m,n);
        v_in = tmp1(:);
        m_in = inside_mask(:);
        v_in_m = v_in(m_in);
        v_in_m(v_in_m == 0) = [];
        
        if isempty(v_in_m)
            v_in_aver = 0;
        else
            v_in_aver = mean(v_in_m);
        end
end
        
        maxVal_loc = 0.95* max(tmp(:));        
        tmp(tmp == 0) = maxVal_loc;
        minVal_loc = 1.05*min(tmp(:));
        if maxVal_loc-minVal_loc > (200 + 0.1*averVal)
            continue;
        end
        if v_in_aver > 0 && ( abs(averVal - v_in_aver) > 150 )
            continue;
        end
    end
    
    SE = ones(ws,ws);
    % croi_d = imdilate_my(croi, ws);
    croi = imdilate(croi, SE, 'same');
    
    d_tmp(minY : maxY , minX : maxX ) = croi(ws_y_s + 1: end-ws_y_e, ws_x_s + 1:end-ws_x_e);
    
    %figure; imagesc(d_tmp)
end

% depth_out_lr = max(d_m, d_tmp);
depth_out_lr = d_m;
for y = 1 : h
    for x = 1 :w
        if depth_out_lr(y,x) == 0
            depth_out_lr(y,x) = d_tmp(y,x);
        end
            
    end
end
depth_out = imresize(depth_out_lr, 1/ resizeFactor, 'nearest');

depth_out_lr( depth_out_lr > 1800) = 1800;
depth_out_lr( depth_out_lr < 0) = 0;
depth_out_lr_f = double(depth_out_lr );
depth_out_lr_n = (depth_out_lr_f./1800.)*255;
depth_out_lr = uint8(depth_out_lr_n);

depth_norm = depth_out;
depth_norm( depth_norm > 1800) = 1800;
depth_norm( depth_norm < 0) = 0;
depth_norm_f = double(depth_norm );
depth_norm_n = (depth_norm_f./1800.)*255;
depth_norm = uint8(depth_norm_n);

% figure; imagesc(depth_out_lr);
end

