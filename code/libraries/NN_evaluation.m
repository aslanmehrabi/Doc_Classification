function [er,Output] = NN_evaluation(net,X,Y)
Output = sim(net,X); %Computation of the network outputs
er  = 100;

%compute the error rate
