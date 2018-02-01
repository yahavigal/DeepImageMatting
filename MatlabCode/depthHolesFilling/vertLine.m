function [depth_out] = vertLine(depth_in, ind)
[ h, ~] = size(depth_in);
depth_out = depth_in;

maxLh1 = 0.1*h;
th1 = 250;
maxLh2 = 0.2*h;
th2 = 50;
% interpolate horizontal line
prev = depth_in(1,ind); 
l1 = 2;
while l1 <= h
    curr = depth_in(l1, ind);
    if curr == 0 && curr ~= prev
        if l1 == h
            depth_out(l1,ind) =  prev ;
            l1 = l1 + 1;
            continue;
        end
        ll = l1 + 1;
        next = depth_in(ll, ind);
        while next == 0 && ll < h
            ll = ll + 1;
            next = depth_in(ll, ind);
        end
        L = ll -l1;
              
        if next ~= 0 &&  (( L < maxLh1 && abs(int32(next) - int32(prev)) < th1) || ( L < maxLh2 && abs(int32(next) - int32(prev)) < th2 ))
            if L ~= 0
                addFactor = round((int32(next) - int32(prev) )/L);
            else
                addFactor = round((int32(next) + int32(prev))/2) ; 
            end
             val = int32(prev);
            for i1 = l1 : ll
                val = val + addFactor;
                depth_out(i1, ind) =  val ;
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