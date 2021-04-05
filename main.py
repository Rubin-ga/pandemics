#!/usr/bin/env python
# coding: utf-8

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import matplotlib
import matplotlib.pyplot as plt
import time
from IPython import display
import numpy as np
import copy

""" A simple model of a pandemics. Agents behave in accordance
with their mental profiles, which are influenced by the neighbourhood.
"""

import agent
import disease as dis
import visualisation as vis

class Config():
    def __init__(self):
        self.group_pressure_inc = 0.05

    def parse_from_file(self, filepath):
        pass

default_config = Config()

class PandemicsModel(Model):
    def __init__(self, N=100, V=5, width=10, height=10,
                 disease=dis.covid_disease, config=default_config):
        self.num_agents = N
        self.disease = disease
        self.deceased = []
        self.buried = []
        self.deceased_counter = 0
        self.infected_counter = 0
        self.grid = MultiGrid(width, height, True)
        self.safety_per_cell = np.ones((height, width))
        self.schedule = SimultaneousActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"deceased": "deceased_counter",
                            "infected": "infected_counter"},
            agent_reporters={"hp": lambda a: a.profile["hp"],
                             "infection_day": lambda a: a.profile["infection_day"],
                            "obedience": lambda a: a.profile["obedience"],
                            "fear": lambda a: a.profile["fear"]}
        )
        self.config = config
        # Create agents
        for i in range(self.num_agents):
            a = agent.create_random_citizen_agent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        for i in self.random.choices(self.schedule.agents, k=V):
            i.start_infection()
        self.running = True
        self.datacollector.collect(self)

    def bury_agent(self, a):
        self.schedule.remove(a)
        self.grid.remove_agent(a)
        self.deceased_counter += 1
        self.buried.append(a)

    def risk_from_agents(self, agents, weight):
        risk = 0
        for a in agents:
            risk += 1 - a.mask_protection
        return risk*weight

    def evaluate_safety_per_cell(self):
        """ 1.0 is a perfectly safe cell - empty, with all neighbours
        and neighbours-of-neighbours empty as well """
        for content, x, y in model.grid.coord_iter():
            self.safety_per_cell[y][x] = 1 # initial value
            # Compute risk from (x,y) cell
            ring0_risk = self.risk_from_agents(content, weight=0.5)
            considered_cells = {(x,y)}

            # Compute risk coming from the neighbours of (x,y) cell
            neighbours = self.grid.get_neighborhood(
                (x,y), moore=True, include_center=False)
            neighbours_content = self.grid.get_cell_list_contents(neighbours)
            ring1_risk = self.risk_from_agents(neighbours_content, 0.25)
            considered_cells | set(neighbours)

            # Compute risk coming from
            # the neighbours of the neighbours of (x,y) cell
            neighbours_of_neighbours = set()
            for c in neighbours:
                neighbours_of_neighbours | set(
                    self.grid.get_neighborhood(
                        (x,y),moore=True, include_center=False))
            neighbours_of_neighbours -= considered_cells
            ring2_risk = self.risk_from_agents(
                self.grid.get_cell_list_contents(neighbours_of_neighbours), 0.125)
            self.safety_per_cell[y][x] -= ring0_risk + ring1_risk + ring2_risk

    def step(self):
        self.evaluate_safety_per_cell()
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        for d in self.deceased:
            self.bury_agent(d)
        self.deceased = []

    def run_model(self, n):
        for i in range(n):
            self.step()


""" Runnig and visualising a simulation """
from matplotlib import gridspec

grid_width = 20
grid_height = 20
agents_count = 20
infected_initially_count = 5

fig_height = grid_height * 0.4
fig_cols = 2
fig_rows = 2

model = PandemicsModel(agents_count, infected_initially_count, grid_width, grid_height)
num_steps = 30
for i in range(num_steps):
    model.step()
    fig = plt.figure(figsize=(fig_height * fig_cols * grid_width/grid_height,fig_height))
    gs = gridspec.GridSpec(fig_rows, fig_cols, width_ratios=[1, 1], height_ratios=[10,1])
    ax0 = fig.add_subplot(gs[0, 0])
    #visualise_agent_position(ax0, model, model.schedule.agents[0])
    vis.visualise_all_agents_position_and_covid_status(ax0, model)
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])
    #vis.heatmap_safety_per_cell(ax1, cax, model)
    vis.heatmap_all_agents_profile_feature(ax1, cax, model, 'obedience')
    display.clear_output(wait=True) # Uncomment to see plt imgs as animation
    display.display(plt.gcf())
    #time.sleep(0.1)
    plt.close(fig)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(
    fig_height * fig_cols * grid_width/grid_height,fig_height))
axes[0, 0].set_ylim([0, model.num_agents])
axes[0, 1].set_ylim([0, model.num_agents])
infected_count = model.datacollector.get_model_vars_dataframe()["infected"]
infected_count.plot(ax=axes[0, 0], title="infected")
deceased_count = model.datacollector.get_model_vars_dataframe()["deceased"]
deceased_count.plot(ax=axes[0, 1], title="deceased")
vis.plot_all_agents_feature(axes[1, 0], model, "fear")
vis.plot_all_agents_feature(axes[1, 1], model, "infection_day")


#for a in model.schedule.agents:
#    plot_agent_feature(a, 'hp')


#for a in model.buried:
#    plot_agent_feature(a, 'hp')
