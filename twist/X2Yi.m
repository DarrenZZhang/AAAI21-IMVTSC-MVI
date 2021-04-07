function Y = X2Yi(X, i)

Y = Trans_Faces(shiftdim(X, i-1));  %N*N*V →V*N*N→N*V*N 

% aaaa = shiftdim(T_tensor,2);   若 T_tensor: a*b*c,则 aaa为 c*a*b
% B = shiftdim(A,n) 将数组 A 的维度移动 n 个位置。
% 当 n 为正整数时，shiftdim 向左移动维度；
% 当 n 为负整数时，向右移动维度。
% 例如，如果 A 是 2×3×4 数组，则 shiftdim(A,2) 返回 4×2×3 数组。