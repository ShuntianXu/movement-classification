function W = initializeWeights(L_in, L_out)

% randomly initializa weights by selecting values from uniform distribution
% in the range [-epsilon_init, epsilon_init]
% L_in and L_out are the number of nodes in the previous and next layers,
% respectively
% Try to implement Glorot Initialisation, He Initialisation, etc.

epsilon_init = sqrt(6/(L_in+L_out));
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
