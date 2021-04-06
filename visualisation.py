#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import copy

# Prepares axis for a grid visualisation
def visualisation_setup(ax, model, title=None):
    ax.set_title(title)
    ax.set_ylim(-0.5, model.grid.height-0.5)
    ax.set_xlim(-0.5, model.grid.width-0.5)
    ax.set_xticks(np.arange(0, model.grid.width, 2))
    ax.set_yticks(np.arange(0, model.grid.height, 2))

# Visualises single feature from profile for all agents
def heatmap_all_agents_profile_feature(ax, cax, model, feature):
    visualisation_setup(ax, model, feature)
    matrix = np.zeros((model.grid.height, model.grid.width))
    for content, x, y in model.grid.coord_iter():
        f_sum = 0
        for a in content:
            f_sum += a.profile[feature]
        if content:
            f_avg = f_sum/len(content)
            matrix[y][x] = f_avg
        else:
            matrix[y][x] = -1
    cmap = copy.copy(plt.cm.get_cmap('seismic'))
    cmap.set_under('black')
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    ax.set_title(feature)
    plt.colorbar(im, ax=ax, cax=cax, orientation="horizontal")

# Visualises safety for all cells
def heatmap_safety_per_cell(ax, cax, model):
    visualisation_setup(ax, model, "safety per cell")
    im = ax.imshow(model.safety_per_cell, vmin=0, vmax=1, cmap='Blues_r', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, cax=cax, orientation="horizontal")

# Visualises building types
def visualise_buildings_map(ax, cax, model):
    visualisation_setup(ax, model, "building map")
    cmap = matplotlib.cm.get_cmap('Blues_r', 100)
    cmap.set_under('lightgrey')
    im = ax.imshow(model.buildings_map,
            vmin=min(model.config.building_tags.values()),
            vmax=max(model.config.building_tags.values()),
            cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, cax=cax, orientation="horizontal")

# Displays single agent position with given color. If `tail` is True
# also show path from previous position.
def visualise_agent_position(ax, model, agent, c='red', tail=False):
    visualisation_setup(ax, model, "position")
    if agent.prev_pos:
        px, py = agent.prev_pos
    else:
        px, py = agent.pos
    x, y = agent.pos
    ax.plot([x], [y], color=c, markersize=5, marker='o')
    if tail and abs(px-x) <= 1 and abs(py-y) <= 1: # the grid is toroidal
        ax.plot([px, x], [py, y], color=c, markersize=5, marker=None)

# Display all agents positions coloring them by colormap
def visualise_all_agents_position(ax, model):
    visualisation_setup(ax, model, "position")
    cmap = plt.cm.get_cmap('summer')
    #normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(model.schedule.agents))
    for a in model.schedule.agents:
        visualise_agent_position(ax, model, a, cmap(norm(a.unique_id)), True)

# Maps agent infection state to color
def get_color_for_infection_state(agent):
    if agent.is_infected() and not agent.is_infectious():
         return "pink"
    elif agent.is_infectious() and not agent.is_symptomatic():
         return "red"
    elif agent.is_infectious() and agent.is_symptomatic():
         return "orange"
    elif agent.is_immune():
         return "green"
    return "yellow"

# Displays all agents positions coloring them by COVID status
def visualise_all_agents_position_and_covid_status(ax, model):
    visualisation_setup(ax, model, "position and COVID status")
    for a in model.schedule.agents:
        c = get_color_for_infection_state(a)
        visualise_agent_position(ax, model, a, c=c, tail=True)

def plot_agent_feature(ax, model, agent, feature):
    all_data = model.datacollector.get_agent_vars_dataframe()
    agent_data = all_data.xs(agent.unique_id, level="AgentID")[feature]
    agent_data.plot(ax=ax)
    ax.set_title(feature)

def plot_all_agents_feature(ax, model, feature):
    for a in model.schedule.agents:
        plot_agent_feature(ax, model, a, feature)

