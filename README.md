###########################|----------------------------|#######################################
###########################| RTBM-conditional |#######################################
###########################|----------------------------|#######################################



Python implementation of the function that generates the conditional probability
of the PDF learned by the RTBM, as depicted here https://arxiv.org/abs/1905.11313.
This function has been added as a class method to the developer version (not public)
of the framework "theta", that can be found here https://github.com/RiemannAI/theta.
It will be integrated in the stable (public) version in future realeases.

For further informations about RTBMs and the theta framework please refer to the
official github page https://github.com/RiemannAI/theta.




########## Requirements

- theta 0.0.1





########## Legend


The code concerning the actual conditional probability function can be found in
main/conditional.py, whereas in main/utils.py are collected some utilities used
for the examples.
The directory examples/ contains a couple of examples of usage of the conditional
function, in particular a gaussian mixture and the multivariate student-t distribution
are considered.