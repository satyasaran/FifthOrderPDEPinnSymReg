# Core libraries
import numpy as np
import random
import time

# Data processing libraries
import pandas as pd
import scipy.io
from scipy.interpolate import griddata

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# Deep learning and optimization libraries
import tensorflow as tf
import deepxde as dde
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# Backend-specific setup for deepxde
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    sin = tf.sin


# General parameters
epochs = 10000
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxx = dde.grad.hessian(dy_x, x, i=0, j=0)
    dy_xxxx = dde.grad.hessian(dy_xx, x,i=0, j=0)
    dy_xxxxx = dde.grad.hessian(dy_xxx, x, i=0, j=0)
    return dy_t + y * dy_x +0.5 * dy_xxx+0.5* dy_xxxxx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

ic = dde.icbc.IC(geomtime, lambda x:  0*np.cos(3*np.pi * x[:, 0:1])+np.exp(-20*x[:, 0:1]**2), lambda _, on_initial: on_initial)


data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=100,  num_boundary=100, num_initial=100,num_test=2000)
lamda=0.01
net = dde.nn.FNN([2] + [25] * 4+ [1], "sin", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=0.0024,loss_weights=[1, 1, lamda])


path_dir='/home/trl102/Dir/MyDailyWork/PDE/DeepXde_PDE/FifthOrderPDE_PinnSymReg/Exp/Train'
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-5, patience=2000)
#model.compile("L-BFGS")
losshistory, train_state =model.train(iterations=20000,display_every=500,model_save_path=path_dir+'/model',callbacks=[early_stopping])
dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_dir)

X = geomtime.random_points(100000)
np.save(path_dir+'/X_array.npy', X)
np.save(path_dir+'/train_array.npy', data.train_points())
np.save(path_dir+'/test_array.npy', data.test_points())
np.save(path_dir+'/bc_initial_array.npy',data.bc_points())

t=np.linspace(0,2,500).reshape(-1,1)
x=np.linspace(-1,1,500).reshape(-1,1)
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))

U_pred = griddata(X, y_pred.flatten(), (xx, tt), method='cubic')
np.save(path_dir+"/U_pred.npy", U_pred)
np.save(path_dir+"/f.npy", f)