# -*- coding: utf-8 -*-
"""Plot_Generator

# Install the requirements
"""

pip install -r requirements.txt

"""# Environment Variables"""

import os

import dotenv

dotenv.load_dotenv('/.env')

os.environ.get('GROQ_KEY')

"""# Tool for interactive plotting"""

import plotly.graph_objects as go
import plotly.express as px
from crewai_tools import tool

@tool
def generalized_plot(*args, plot_type, title, xlabel, ylabel):
    """
    Plots the given variables.

    Parameters:
    - *args: Any number of variables (lists/arrays) to be plotted.
    - plot_type (str): Type of plot ('line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'bubble', 'heatmap', '3d_scatter', '3d_line', '3d_surface')
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    """
    fig = go.Figure()

    if plot_type in ['line', 'scatter', 'bubble', '3d_scatter', '3d_line']:
        for i, arg in enumerate(args):
            x = list(range(len(arg)))
            name = f'Variable {i+1}'
            if plot_type == 'line':
                fig.add_trace(go.Scatter(x=x, y=arg, mode='lines', name=name))
            elif plot_type == 'scatter':
                fig.add_trace(go.Scatter(x=x, y=arg, mode='markers', name=name))
            elif plot_type == 'bubble':
                fig.add_trace(go.Scatter(x=x, y=arg, mode='markers', marker=dict(size=[20]*len(arg)), name=name))
            elif plot_type == '3d_scatter':
                fig.add_trace(go.Scatter3d(x=x, y=arg, z=[0]*len(arg), mode='markers', name=name))
            elif plot_type == '3d_line':
                fig.add_trace(go.Scatter3d(x=x, y=arg, z=[0]*len(arg), mode='lines', name=name))

    elif plot_type == 'bar':
        for i, arg in enumerate(args):
            fig.add_trace(go.Bar(x=list(range(len(arg))), y=arg, name=f'Variable {i+1}'))

    elif plot_type == 'histogram':
        for i, arg in enumerate(args):
            fig.add_trace(go.Histogram(x=arg, name=f'Variable {i+1}'))

    elif plot_type == 'box':
        for i, arg in enumerate(args):
            fig.add_trace(go.Box(y=arg, name=f'Variable {i+1}'))

    elif plot_type == 'violin':
        for i, arg in enumerate(args):
            fig.add_trace(go.Violin(y=arg, name=f'Variable {i+1}'))

    elif plot_type == 'pie':
        if len(args) == 1:
            fig.add_trace(go.Pie(labels=[f'Category {i+1}' for i in range(len(args[0]))], values=args[0]))
        else:
            raise ValueError("Pie plot requires exactly one variable (list/array).")

    elif plot_type == 'heatmap':
        if len(args) == 1:
            fig.add_trace(go.Heatmap(z=args[0]))
        else:
            raise ValueError("Heatmap plot requires exactly one variable (2D list/array).")

    elif plot_type == '3d_surface':
        if len(args) == 1:
            fig.add_trace(go.Surface(z=args[0]))
        else:
            raise ValueError("3D Surface plot requires exactly one variable (2D list/array).")

    else:
        raise ValueError("Unsupported plot_type.")

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white'
    )

    fig.show()

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white'
    )

    fig.show()

    return fig.show()

"""# Enter User Query Here:"""

user_query = input("Enter your query")

"""# Data Analysis"""

path = input("Enter the path of the dataset here: ")
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv,dotenv_values
from langchain_experimental.tools import PythonREPLTool
import pandas as pd


load_dotenv()

import pandas as pd
df = pd.read_csv(path)
data = df.head()
print(data)
llm = ChatGroq(temperature=0, groq_api_key=os.environ.get('GROQ_KEY'), model_name="llama3-70b-8192")


evaluator_1 = Agent(
    role='Read the dataset',
    goal=f"""
    You are given with the dataset path. You should read the dataset and tell the semantic meaning of it.
    Write a python code and execute.
    Next you should point out the numerical, categorical variables separately.
    Next you should answer the user question about the dataset.. Use tool to get the answer from the dataset.
    the user query is here: {user_query}
    For running python code you have repl tool, while performing action your key should be query and value should be your code you generated for the process
    data set path: {path}

    eg:
    The dataset is about ...
    categorical: c1,c2...
    numerical: c3,c5..
    describe each column's meaning like
    c1 is about ...
    """,
    backstory="You are an expert in analysing datasets",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools = [PythonREPLTool()]
    )

evaluator_2 = Agent(
    role='Perform statistical analysis',
    goal=f"""
    Write a python code,
    Based on the dataset details. You should analyse the dataset statistically. Like find mean, median, mode, etc.
    the output should be a statistical analysis report with semantics and inferential result of statistical analysis.
    For running python code you have repl tool, while performing action your key should be 'query' and value should be your code you generated for the process
    the path of the dataset is here: {path}, use it when generating and executing code
    """,
    backstory="You are statistician with sharp data analysis skills, based on provided columns u perform statistical analysis",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools = [PythonREPLTool()]
    )

plot_agent = Agent(
    role='Perform Data Visualization',
    goal=f"""
    You have the dataset path, you have python REPL tool,
    For running python code you have repl tool, while performing action your key should be 'query' and value should be your code you generated for the process
    You have to generate plots for the dataset and execute them (generate code and execute it)
    save the plots in the /content/plots folder.
    DONT GET STUCK IN THE EXECUTION LOOP PRODUCE NECESSARY PLOTS ONLY!
    Generate just around 10 plots.
    dataset path is here: {path}
    """,
    backstory="You are a statistical analyst, analysing the dataset and plotting them to infer results",
    verbrose = True,
    allow_delegation=False,
    llm=llm,
    tools = [PythonREPLTool()]

)

eval_1 = Task(
    description="Read the dataset and describe it. Use the tool to get info about dataset and generate and execute the code with the given tools",
    agent=evaluator_1,
    expected_output="The result must be a analysis of the dataset and produce the statistical figures"
    )

eval_2 = Task(
    description="Perform statistical analysis on the dataset. Use the tool to get info about dataset and generate and execute the code with the given tools",
    agent=evaluator_2,
    expected_output="Final result must be statistical analysis report with semantics of the dataset, like what the dataset is about"
    )

task_3 = Task(
    description="Produce the plots of the dataset. generate and execute the code with the given tools, generate plots. Use the same dataset. (matplotlib library is preferrable for plotting you can use others too)",
    agent=plot_agent,
    expected_output="Final output should be relevant plots describing the dataset"
)



crew = Crew(
    agents=[evaluator_1,evaluator_2,plot_agent],
    tasks=[eval_1,eval_2,task_3],
    verbose=2,
    process=Process.sequential,
    )

result = crew.kickoff({"path":path,"user_query":user_query})
print(result)



