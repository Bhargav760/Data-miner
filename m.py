import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
d=pd.read_csv("iris.csv")

def fun(input):
    file = pd.read_csv(input.name)
    return file
def findmean(col):
    sum = np.sum(np.array(d.loc[:,col]))
    return sum/len(d)

def scatterplot(d1,d2):
    x=np.array(d.loc[:,d1])
    y=np.array(d.loc[:,d2])
    fig = plt.figure()
    plt.scatter(x,y)
    return fig

with gr.Blocks() as demo:
    with gr.Tab("Display dataset"):
        dataset = gr.File(label="File")
        output = gr.Dataframe(label="Dataset")
        btn = gr.Button(label ="Submit")
        btn.click(fn=fun, inputs = dataset, outputs = output)
    with gr.Tab("Find Mean"):
        dd=gr.Dropdown(list(d.columns), label="Select Column")
        mean = gr.TextArea(label="Mean")
        btn=gr.Button("Submit")
        btn.click(findmean,dd,mean)
    with gr.Tab("Scatter plot"):
        dd=gr.Dropdown(list(d.columns), label="Select Column")
        ddd=gr.Dropdown(list(d.columns), label="Select Column")
        plot = gr.Plot(label="graph")
        btn=gr.Button("Submit")
        btn.click(scatterplot,inputs =[dd,ddd],outputs=plot)
        
demo.launch()