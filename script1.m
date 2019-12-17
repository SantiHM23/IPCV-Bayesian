clear all; clc;

inputdata = load('./template_data/data/data.mat');

for i = 1:size(inputdata.labels,1)
    if inputdata.labels(i) == 8
        inputdata.labels(i) = 4;
    end
end

mu_v = zeros(1,240);
sigma_v = zeros(1,240);

for j = 1:4
    counter = 1;
    m = [];
    for k = 1:size(inputdata.labels,1)
        if inputdata.labels(k) == j
            for p = 1:3
                m(:,p,counter) = inputdata.data(:,p,k); 
            end
            counter = counter + 1;
        end
    end
    [mu, sigma] = fit_gaussian(m);
    mu_v(1+60*(j-1):60*j) = mu;
    sigma_v(1+60*(j-1):60*j) = sigma;
end
