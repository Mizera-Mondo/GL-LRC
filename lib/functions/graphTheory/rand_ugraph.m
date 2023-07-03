function A = rand_ugraph(nv,ne,dw,rw)
% Generate a random undirected simple graph of nv vectices and ne edges
% nv: Number of vetices
% ne: Number of edges
% dw: Default value of edge weight
% rw: Edge weight random tuning parameter, 0 for no tuning. rw < 2

    A = zeros(nv); % Initialize adjacency
    order = 0;
    if nv == 1
        A = 0;
    elseif nv > 1
        e_max = (1+(nv-1))*(nv-1)/2;
        if ne > e_max
            error('too many edges');
        else
            edges = randperm(e_max,ne); % Random array determining order of chosen edges
            edges = sort(edges);
            for j = 2:nv
                for i = 1:j-1
                    order = order + 1;
                    %disp(order);
                    if ismember(order,edges) == 1 % If connected, construct adjacency
                        A(i,j) = dw + rw*dw*(rand - 0.5);
                        A(j,i) = A(i,j);
                    end
                end
            end
        end
     end
 end

