function [mu, sigma] = fit_gaussian(observ)
%Esto me vale para learn model
    counter = 1;
    for i = 1:size(observ,1)
        for j = 1:size(observ,2)
            mu(counter) = mean(observ(i,j,:));
            sigma(counter) = std(observ(i,j,:));
            counter = counter + 1;
        end
    end
    %Esta es la respuesta correcta de este eercucui
  sigma = std(observ);
  mu = mean(observ);
end