# Importing required libraries
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Importing DeepXDE library for solving partial differential equations using deep learning
import deepxde as dde

# Importing Scikit-Optimize library for Bayesian hyperparameter optimization
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import joblib
from skopt import dump, load


if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf

    sin = tf.sin

# General parameters
epochs = 20000
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxx = dde.grad.hessian(dy_x, x, i=0, j=0)
    dy_xxxx = dde.grad.hessian(dy_xx, x,i=0, j=0)
    dy_xxxxx = dde.grad.hessian(dy_xxx, x, i=0, j=0)
    return dy_t + y * dy_x +0.5 * dy_xxx+0.5 * dy_xxxxx


def create_model(config):
    learning_rate, num_dense_layers, num_dense_nodes, activation,lamda = config

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, lambda x: -1, lambda _, on_boundary: on_boundary)
 
    ic = dde.icbc.IC(geomtime, lambda x:  0*np.cos(3*np.pi * x[:, 0:1])+np.exp(-20*x[:, 0:1]**2), lambda _, on_initial: on_initial
)
    #data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=1500,  num_boundary=100, num_initial=100,num_test=3000)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=100, num_boundary=100, num_initial=100, num_test=200)
    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        "Glorot uniform",
    )


    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate,loss_weights=[1, 1, lamda])
    return model

def train_model(model, config):
    #early_stopping = dde.callbacks.EarlyStopping( monitor="loss_train", min_delta=1e-8, patience=2000)
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=2000)
    #losshistory, train_state = model.train(iterations=10000,display_every=100,callbacks=[early_stopping,checkpointer])   
    losshistory, train_state =model.train(iterations=20000,display_every=2000,callbacks=[early_stopping])

    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    steps=np.array(losshistory.steps)[:, None]

    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    #skopt.dump(train, 'Nomagnetic_5th_6'+str(ITERATION)+ "/train.pkl")
    #skopt.dump(test, 'Nomagnetic_5th_6'+str(ITERATION)+ "/test.pkl")
    #skopt.dump(steps, 'Nomagnetic_5th_6'+str(ITERATION)+ "/steps.pkl")


    error = test.min()
    return error




# HPO setting
n_calls = 15
dim_learning_rate = Real(low=1e-4, high=5e-1, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=1, high=30, name="num_dense_nodes")
dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")
lamda = Real(low=0.01, high=0.1, name="lamda", prior="log-uniform")


dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    lamda
]


default_parameters = [1e-2, 1,10, "sigmoid",0.1]

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation,lamda):

    config = [learning_rate, num_dense_layers, num_dense_nodes, activation,lamda]
    global ITERATION

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print("lamda:", lamda)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(config)
    # possibility to change where we save
    error = train_model(model, config)
    # print(accuracy, 'accuracy is')

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error


ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

print(search_result.x)
#print(search_result)

# Save the search_result object
dump(search_result, 'results_comb_gp4_a100_exp.pkl', store_objective=False)

# Load the search_result object in the future
#loaded_search_result = joblib.load('search_result.pkl')

#plot_convergence(search_result)

#plot_objective(search_result, show_points=True, size=3.8)
