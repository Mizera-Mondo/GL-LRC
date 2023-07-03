function A = rand_digraph(nv,ne,dw,rw)
%% Generate a random directed simple graph of nv vectices and ne edges
% nv: Number of vetices
% ne: Number of edges
% dw: Default value of edge weight
% rw: Edge weight random tuning parameter, 0 for no tuning. rw < 2

    A = zeros(nv,nv); % Initialize adjacency
    order = 0;
    if nv == 1
        A = 0;
    elseif nv > 1
        e_max = nv*(nv-1);
        if ne > e_max
            error('too many edges');
        else
            edges = randperm(e_max,ne); % Random array determining order of chosen edges
            edges = sort(edges);
            for j = 1:nv
                for i = [1+(j==1):j-1 j+1:nv]
                    order = order + 1;
                    if ismember(order,edges) == 1 % If connected, construct adjacency   
                        A(j,i) = dw + rw*dw*(rand - 0.5);
                    end
                end
            end
        end
     end
 end

