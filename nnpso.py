import torch
import torch.nn as nn
import tqdm
import copy
import numpy as np
import pandas as pd

def create_velocity(model):
  """
  Returns random velocity for the particle
  Args:
    model: The model which will be trained
  Returns:
    velocity: randomly initialized list of tensors with tensors of model layers shapes
  """
  velocity = []
  for m in model:
    if isinstance(m, nn.Linear):
      velocity.append(torch.rand_like(m.weight))
  return velocity

def update_velocity(model,velocity,global_best,local_best,alpha=1,beta=1,theta=0.7):
  """
  Returns updated velocity for the particle
  Args:
    model: the model which will be trained
    velocity: the curent particle velocity
    global_best: the paritcles` postion with the best metric
    local_best: the paritcle`s with the best metric
    alpha: float number, represents how paritcle tends to global_best postion
    beta: float number, represents how paritcle tends to local_best postion
    theta: float number, represents inertia parameter
  Returns:
    new_velocity: a list of tensors with tensors of model layers shapes
  """
  new_velocity = []
  j=0
  for i in range(len(model)):
    if isinstance(model[i], nn.Linear):
      new_velocity.append(theta*velocity[j]+alpha*np.random.rand()*(global_best[i].weight-model[i].weight)+np.random.rand()*beta*(local_best[i].weight-model[i].weight))
      j+=1
  return new_velocity

def update_position(model,velocity):
  """
  Returns updated position for the particle
  Args:
    model: the particle which will be trained
    velocity: the curent particle velocity
  Returns:
    model
  """
  j=0
  for i in range(len(model)):
    if isinstance(model[i], nn.Linear):
      model[i].weight = nn.Parameter(model[i].weight+velocity[j])
      j+=1
  return model

def create_particle(model):
  """
  Returns updated position for the particle
  Args:
    model: the particle which will be trained
    velocity: the curent particle velocity
  Returns:
    model
  """
  new_model = copy.deepcopy(model)
  for i in range(len(new_model)):
    if isinstance(new_model[i], nn.Linear):
      nn.init.uniform_(new_model[i].weight,a=-1,b=1)
  return new_model

def train(model,metric,data,labels,iterations=100,n_particles=50,alpha=1,beta=1,theta_function=lambda iter: 0.7,minim=True,rand_seed=42):
  """
  Returns updated position for the particle
  Args:
    model: the particle which will be trained
    metric: the curent particle velocity
    data: features
    labels: targets
    iterations: number of iterations
    n_particles: number of particles
    alpha: float number, represents how paritcle tends to global_best postion
    beta: float number, represents how paritcle tends to local_best postion
    theta_function: function that takes curent iteration and returns inertia
    minim: True if solving minimisation, False if maximisation
    rand_seed: random seed
  Returns:
    global_best: model with the best performance
    history: history of particles results
    global_best_hist: history of global_best results
  """
  np.random.seed(rand_seed)
  particles = [create_particle(model) for _ in range(n_particles)]
  global_best = None
  global_best_y = None
  global_best_hist=[]
  local_bests = [None for _ in range(n_particles)]
  local_bests_y = [None for _ in range(n_particles)]
  velocities = [create_velocity(model) for _ in range(n_particles) ]
  history = []
  for iter in tqdm.tqdm(range(iterations)):
    results = [metric(particles[i](data),labels).detach().numpy().item() for i in range(n_particles)]
    for i in range(n_particles):
      if global_best_y is None or (global_best_y > results[i] and minim) or (global_best_y < results[i] and not minim):
        global_best_y = results[i]
        global_best = copy.deepcopy(particles[i])
      if local_bests_y[i] is None or (local_bests_y[i] > results[i] and minim) or (local_bests_y[i] < results[i] and not minim):
        local_bests_y[i] = results[i]
        local_bests[i] = copy.deepcopy(particles[i])
    for i in range(n_particles):
      particles[i] = update_position(particles[i],velocities[i])
      velocities[i] = update_velocity(particles[i],velocities[i],global_best,local_bests[i],alpha=alpha,beta=beta,theta=theta_function(iter))
    history.append(copy.deepcopy(results))
    #print(results)
    global_best_hist.append(global_best_y)
  return global_best,pd.DataFrame(history),global_best_hist