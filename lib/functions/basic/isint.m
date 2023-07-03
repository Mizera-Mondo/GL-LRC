function flag = isint(n)
%ISINT judge whether the input n is an integer.
%   flag: logical
flag = n - round(n) == 0;
end

