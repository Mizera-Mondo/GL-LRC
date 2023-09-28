function [A, M] = random_geometric_graph(n, d, r)
    % 生成n个d维随机点
    points = rand(n, d);
    
    % 计算距离矩阵
    M = squareform(pdist(points));
    
    % 创建邻接矩阵
    A = M <= r;
    A = A - diag(diag(A)); % 将对角线元素设置为0
end