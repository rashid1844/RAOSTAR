#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import sys
from utils import import_models
import_models()
from hybrid_rover_model import *
from raostar import RAOStar
import graph_to_json
import time
import copy
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from shapely.geometry import *
import matplotlib
from matplotlib.collections import PatchCollection
from descartes import *

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
from scipy.stats import multivariate_normal


def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_contour(x,y,z):
    # define grid.
    xi = np.linspace(min(x), max(x), 20)
    yi = np.linspace(min(y), max(y), 20)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    #plt.colorbar() # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    return plt

def extract_policy(graph,obstacles,goal_region):
    # extract policy mapping nodes to actions
    # self.debug("===========================")
    # self.debug("=== Extract Policy ========")
    # self.debug("===========================")
    queue = deque([graph.root])  # from root
    policy = {}
    k=0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    risk = 0
    mu_total = np.array([[-999,-999]])
    while len(queue) > 0:
        node = queue.popleft()
        state = list(node.state.belief.keys())[0]
        mu = state[0].get_mu()
        sigma = state[0].get_sigma()
        mu = np.squeeze(np.asarray(mu))
        sigma = np.squeeze(np.asarray(sigma))

        
        mu_current = np.array([[mu[0],mu[1]]])
        mu_total = np.concatenate((mu_total,mu_current),axis=0)
        size_before = np.size(mu_total,axis=0)
        mu_total = np.unique(mu_total,axis=0)
        size_after = np.size(mu_total,axis=0)

        # x = uniform(float(mu[0])-2, float(mu[0])+2, 200)
        # y = uniform(float(mu[1])-2, float(mu[1])+2, 200)
        # z = gauss(x, y, sigma[0:2,0:2], mu[0:2])
        # plt = plot_contour(x,y,z)

        x, y = np.mgrid[float(mu[0])-1.5:float(mu[0])+1.5:.1, float(mu[1])-1.5:float(mu[1])+1.5:.1]
        pos = np.dstack((x, y))
        print("---------------")
        if node.best_action!=None:
            print(node.best_action.name)
        else:
            print(node.best_action)
            
        print(mu[0:2])
        print(state[1])
        print(node.risk)
        print(node.likelihood)
        print("---------------")
        rv = multivariate_normal(mu[0:2], sigma[0:2,0:2])
        if size_after == size_before:
            ax.contour(x, y, rv.pdf(pos))
        risk = risk + node.risk
        if node.best_action != None:
            policy[(node.name,node.probability,node.depth)] = node.best_action.name
            children = graph.hyperedges[node][node.best_action]
            for c in children:
                queue.append(c)
                k=k+1

    print("risk: "+str(risk))

    BLUE = '#0000ff'
    RED = '#ff0000'
    GREEN = '#008000'

    obs = Polygon(obstacles)
    obs_poly = PolygonPatch(obs, fc=RED, ec=RED, alpha=0.5, zorder=2)

    goal = Polygon(goal_region)
    goal_poly = PolygonPatch(goal, fc=GREEN, ec=GREEN, alpha=0.5, zorder=2)
    ax.add_patch(obs_poly)
    ax.add_patch(goal_poly)
    ax.set_xlim([-1,12])
    ax.set_ylim([-1,12])
    plt.show()
                
    return policy

    

# Now you can give command line cc argument after filename
if __name__ == '__main__':
    # default chance constraint value
    cc = 0.2
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

    obstacles = np.array([[3,3],[4,3],[4,4],[3,4]])
    # obstacles = np.array([[4,4],[3,5],[1,3],[2,3]])

    goal = Polygon([[8,8],[10,8],[10,10],[8,10]])
    # goal = Polygon([[8,2],[10,2],[10,4],[8,4]])
    
    model = HybridRoverModel(goal_region=goal, obstacle = obstacles, goal_num_sample=3, prob_sample_success=0.95, DetermObs=True)

    algo = RAOStar(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = 13, random_node_selection=False)
    # algo = RAOStar(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = 3, random_node_selection=False, time_limit=60*45)

    sigma_b0 = 0.01
    n = 4

    initial_mu = np.zeros((2*n,1))
    initial_sigma = np.zeros((2*n,2*n))
    initial_sigma[0:4,0:4] = sigma_b0*np.eye(n)

    initial_gaussian_state = GaussianState(mu=initial_mu, sigma=initial_sigma)

    b_init = {(initial_gaussian_state, 0): 1.0}
    P, G = algo.search(b_init)
    print(algo.graph.root.exec_risk)
    print(algo.graph.root.value)

    # print("Root risk : ",algo.graph.root.exec_risk)
    # print("Root value: ",algo.graph.root.value)

    algo.extract_policy()
    print(P)
    extract_policy(algo.graph,obstacles,goal)


