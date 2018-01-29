function [im_d] = imdilate_my(im,se_size)
% my implementation as Matlab function looks to have a bug
im_d = im;
[h,w] = size(im);
sz = floor(se_size/2);
for y = sz + 1 : h - sz
    for x = sz + 1 : w -sz
        crop = im(y - sz : y + sz, x - sz : x + sz);
        im_d(y,x) = max(crop(:));
    end   
end

end

