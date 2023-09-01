# Machine learning models for the prediction of fibre-reinforced concrete behaviour

As it gets error when pushing other repository, please find the main part of this research in the Physics-driven machine learning models repository: https://github.com/adofarsi/fibre-reinforced-concrete-ml.git MengruiHan's branch

This project combines the Firedrake framework with machine learning (ML) techniques to conduct simulations of fiber-reinforced concrete (FRC) structures. Within Firedrake, a linear elastic forward model for a representative volume and a three-point bending test model using a linear-elastic constitutive relationship were constructed to simulate the initial elastic phase of concrete under stress. Developed machine learning models to reproduce the forward model and select the best one for further use. Subsequently, following a similar workflow, ML models are developed to replace the cohesive crack model using the discontinuous Galerkin method described in Wei's paper, aiming to predict the appearance of cracks in concrete as stress increases. Finally, both the numerical and ML models are integrated into the physic-driven ML interface designed by Dr. Nacime, expanding its applicability and facilitating future optimization efforts. The research results offer insights into improving the simulation workflow.  As the cohesive crack model has a high time cost and propensity for non-convergence in numerical model predictions, the adoption of the ML model significantly reduces the simulation costs. Notably, during the linear elastic phase, the use of ML also took into account the physical error derived from the model, enhancing the robustness and authenticity of the predictions. By considering both the error from the physical model and the inherent error from ML predictions, the reliability of the model is heightened.


## Table of Contents
* [final final workflow in notebook]
* [physics_driven_ml]


## Generate dataset

* [generate_linear_data]

Data stored in data/datasets/linear_data

* [generate_three_point_bending_data]

Data stored in data/datasets/linear_three_point

* [generate_cohesive_crack_model]

Data stored in data/datasets/cohesive_crack_data
Simulation model stored in dataset_processing/train_data


## Training

* [train_linear_elasticity]
* [train_phy_linear]
* [train_cohesive_crack_model]

ML model stored in physics_driven_ml/model
The best trainning model stored in saved_models
