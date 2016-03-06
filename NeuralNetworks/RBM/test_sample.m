function test_sample(probabilities)
sum = 0;
for i=1:10000
    binary = sample_bernoulli(probabilities);
    sum = binary + sum;
end
    disp('sum is');
    disp(sum);
end