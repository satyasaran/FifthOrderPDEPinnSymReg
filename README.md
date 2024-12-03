Physics of Fluids: Under revision
Authors: Satyasaran Changdar, Bivas Bhaumik, Nabanita Sadhukhan, Sumit Pandey, Sabyasachi Mukhopadhyay, Soumen De, and Serafim Bakalis.
Abstract: 
This study explores a hybrid framework integrating machine learning techniques
and symbolic regression via genetic programming for analyzing the nonlinear 
propagation of waves in arterial blood flow. We employ a mathematical framework to
simulate viscoelastic arterial flow, incorporating assumptions of long wavelength
and large Reynolds numbers. We used a fifth-order nonlinear evolutionary equa-
tion using reductive perturbation to represent the behavior of nonlinear waves in a
viscoelastic tube, considering the tube wall’s bending. We obtain solutions through
physics-informed neural networks (PINNs) that optimizes via Bayesian hyperpa-
rameter optimization across three distinct initial conditions. We found that physics-
informed neural network-based models are proficient at predicting the solutions of
higher-order nonlinear partial differential equations in the spatial-temporal domain
[−1, 1]×[0, 2]. This is evidenced by graphical results and a residual validation show-
ing a mean absolute residue error of O(10^-3). We thoroughly examine the impacts
of various initial conditions. Furthermore, the three solutions are combined into a
single model using the random forest machine learning algorithm, achieving an im-
pressive accuracy of 99% on the testing dataset and compared with another model
using an artificial neural network. Finally, the analytical form of the solutions is
estimated using symbolic regression that provides interpretable models with mean
square error of O(10^-3). These insights contribute to the interpretation of cardio-
vascular parameters, potentially advancing machine learning applications within the
medical domain.
