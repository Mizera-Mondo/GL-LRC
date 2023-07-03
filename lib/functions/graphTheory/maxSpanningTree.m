function T = maxSpanningTree(G)
% MAXSPANNINGTREE - Finds the maximum spanning tree of a connected graph
%   
% Input:
%   G - NxN weighted adjacency matrix of the graph
%
% Output:
%   T - NxN weighted adjacency matrix of the maximum spanning tree

[n,~] = size(G);
T = zeros(n); % Initialize the result matrix

% Create a list of all edges in G, sorted by weight in descending order
edges = [];
for i = 1:n
    for j = i+1:n
        if G(i,j) ~= 0
            edges = [edges; i j G(i,j)];
        end
    end
end
edges = sortrows(edges, -3); % Sort the edges by weight in descending order

% Initialize the parent array for each vertex
global parents
parents = 1:n;

% Loop through all the edges, adding each edge to the tree if it doesn't form a cycle
i = 1;
e = 0;
while e < n-1
    edge = edges(i,:);
    i = i + 1;
    x = edge(1);
    y = edge(2);
    w = edge(3);
    
    root1 = findRoot(x);
    root2 = findRoot(y);
    
    if root1 ~= root2
        e = e + 1;
        T(x,y) = w;
        T(y,x) = w;
        parents(root2) = root1;
    end
end

end

function root = findRoot(x)
    global parents
    pathList = [];

    while parents(x) ~= x
        pathList = [pathList, x];
        x = parents(x);
    end

    parents(pathList) = x; % path compression in the union-find algorithm
    root = x;
end
% function root = findRoot(x, parents)
% % FINDROOT - Finds the root of the set containing x
% %   
% % Input:
% %   x - The vertex for which to find the root
% %   parents - An array containing the parent of each vertex
% %
% % Output:
% %   root - The root of the set containing x
% 
% if parents(x) == x
%     root = x;
% else
%     root = findRoot(parents(x), parents);
%     parents(x) = root;
% end
% 
% end