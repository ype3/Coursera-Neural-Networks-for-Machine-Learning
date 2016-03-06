function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    case_num = size(visible_data, 2);
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_states = sample_bernoulli(hidden_probability);
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_states);
    visible_states = sample_bernoulli(visible_probability);%reconstruction
    %hidden_probability2 = visible_state_to_hidden_probabilities(rbm_w, visible_states);
    
    hidden_probability2 = visible_state_to_hidden_probabilities(rbm_w, visible_states);
    improve = configuration_goodness_gradient(visible_states, hidden_probability2);
    %sampled hidden states
     %hidden_states2 = sample_bernoulli(hidden_probability2);
    %ret =hidden_states * visible_data'  -  hidden_states2 * visible_states';
    ret = hidden_states * visible_data'/case_num - improve';
    %ret = ret/case_num;
end
