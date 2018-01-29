function [depth_out] = horLine(depth_in, ind)
[ ~, w] = size(depth_in);
depth_out = depth_in;

maxLw1 = 0.15*w;
maxLw2 = 0.25*w;
maxLw3 = 0.75*w;
th1 = 300;
th2 = 250;
th3 = 200;

if ind == 1
    maxLw1 = 0.2*w;
    maxLw2 = 0.;
    maxLw3 = 0.;
    th1 = 150;
    th2 =0;
    th3 = 0;
end
% interpolate horizontal line
prev = depth_in(ind,1); 
l1 = 2;
while l1 <= w
    curr = depth_in(ind,l1);
    if curr == 0 && curr ~= prev
        if l1 == w
            depth_out(ind,l1) =  prev ;
            l1 = l1 + 1;
            continue;
        end
        ll = l1 + 1;
        next = depth_in(ind,ll);
        while next == 0 && ll < w
            ll = ll + 1;
            next = depth_in(ind,ll);
        end
        L = ll -l1;
              
        if next ~= 0 && (( L < maxLw1 && abs(int32(next) - int32(prev)) < th1) || ( L < maxLw2 && abs(int32(next) - int32(prev)) < th2) || ( L < maxLw3 && abs(int32(next) - int32(prev)) < th3))
            if L ~= 0
                addFactor = round((int32(next) - int32(prev) )/L);
            else
                addFactor = round((int32(next) + int32(prev))/2) ; 
            end
            val = int32(prev);
            for i1 = l1 : ll
                val = val + addFactor;
                depth_out(ind,i1) =  val ;
            end            
        end
        l1 = ll+1;
        prev = next;
    else % curr == 0 && curr ~= prev
        prev = curr;
        l1 = l1 + 1;
    end   
end
end

