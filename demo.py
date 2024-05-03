import gradio as gr
import nnpso
import torchvision
from sklearn import datasets
import torch
import torch.nn as nn
from math import tanh
from torchmetrics import Accuracy
from matplotlib import pyplot as plt
from functools import partial




def inertia_function0(iteration,max_iter=100,min_theta=0.3,max_theta=1):
  return max_theta

def inertia_function1(iteration,max_iter=100,min_theta=0.3,max_theta=1):
  return min_theta + tanh(iteration * (max_theta-min_theta)/max_iter)

def inertia_function2(iteration,max_iter=100,min_theta=0.3,max_theta=1):
  return max_theta + (min_theta-max_theta)*(max_iter-iteration-1)/(iteration+1)

def train(dataset,iteration_max,num_particles,alpha,beta,inertia,theta_start,theta_end):
    dataset_collection = {
    'FashionMNIST':torchvision.datasets.FashionMNIST('./data', train=True, download=True),
    'MNIST':torchvision.datasets.MNIST('./data', train=True, download=True),
    'iris':datasets.load_iris()
    }
    ds = dataset_collection[dataset]
    
    if dataset=='iris':
        X = ds.data
        y = ds.target
        X = torch.tensor(X).type(torch.float32)
        y = torch.tensor(y).type(torch.float32)
        model = nn.Sequential(
            nn.Linear(4,128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128,64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64,3,dtype=torch.float32),
            nn.Softmax(1)
        )
        n_classes = 3
    else:
        X = ds.data.type(torch.float32)
        y= ds.targets
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128,64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64,10,dtype=torch.float32),
            nn.Softmax(1)
        )
        n_classes = 10
    inertia_function = None
    if inertia == 'constant':
       inertia_function = inertia_function0
    elif inertia =='θ_start+tanh(iteration*(θ_end-θ_start)/iteration_max)':
       inertia_function = inertia_function1
    elif inertia == 'θ_start+(θ_start-θ_end)*(iteration_max-iteration-1)/(iteration+1)':
       inertia_function = inertia_function2

    inertia_function = partial(inertia_function,max_iter=iteration_max,min_theta=theta_start,max_theta=theta_end)

    best_model,history,global_best_history = nnpso.train(
    model=model,
    metric=Accuracy(task="multiclass", num_classes=n_classes),
    data=X,
    labels=y,
    iterations=iteration_max,n_particles=num_particles,
    alpha=alpha,beta=beta,minim=False,theta_function=inertia_function)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    history.plot(ax=ax)
    ax.plot(global_best_history,'--')
    return fig
    #dadsa



    

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Neural networks learned by PSO
    """)
    dataset = gr.Radio(['iris','MNIST','FashionMNIST'],label = 'dataset')
    iteration_max = gr.Number(label='iteration_max',precision=0)
    num_particles = gr.Number(label='number of particles',precision=0)
    alpha = gr.Number(label='alpha')
    beta = gr.Number(label='beta')
    inertia = gr.Radio(['constant',
                            'θ_start+tanh(iteration*(θ_end-θ_start)/iteration_max)',
                            'θ_start+(θ_start-θ_end)*(iteration_max-iteration-1)/(iteration+1)'],
                            label = 'inertia function')
    theta_start = gr.Number(label='θ_start')
    theta_end = gr.Number(label='θ_end')
    plot = gr.Plot(label="learning history")
    run_b = gr.Button('Run')
    run_b.click(train,
                inputs=[dataset,iteration_max,num_particles,alpha,beta,inertia,theta_start,theta_end],
                outputs=plot
    )
        
    
demo.launch()
