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
from collections import defaultdict

import random # for random buildings

""" A simple model of a pandemics. Agents behave in accordance
with their mental profiles, which are influenced by the neighbourhood.
"""

import agent
import disease as dis
import visualisation as vis

HOURS_PER_DAY = 24

class Config():
    def __init__(self, filepath=None, citizens_count=None, width=None, height=None):
        if filepath:
            pass
        else:
            self.width = 50
            self.height = 50
            self.citizens_count = 30
            self.policemen_count = 0
            self.infected_count = 5

            self.steps_per_day = 48

            self.citizens_mental_features_distribution = {
                    # feature:(mean, sd of normal distribution)
                    "fear": (0.5, 0.5),
                    "awareness": (0.5, 0.5),
                    "obedience": (0.5, 0.5)
                    }

            self.policemen_mental_features_distribution = {
                    # feature:(mean, sd of normal distribution)
                    "awareness": (0.75, 0.25),
                    }

            self.group_pressure_inc = 0.10 / self.steps_per_day
            self.ticket_impact = 0.75

        # override if parameters are given
        if width:
            self.width = width
        if height:
            self.height = height
        if citizens_count:
            self.citizens_count = citizens_count

        self.building_tags = {"house": 1, "workplace": 2, "shop": 3}

        self.day_plan = {
                0: 'house',
                7: None,
                8: 'workplace',
                16: None,
                17: 'shop',
                19: None,
                20: 'house',
                }


        self.parse_buildings()
        #self.create_random_buildings(minimum_margin=1)

    def parse_buildings(self):
        # TODO: read description from a file
        self.houses_count = 5
        self.workplaces_count = 3
        self.shops_count = 5

        houses_sector = (0, self.width // 5) # Size proportions: house - 1, workplace - 2, shop - 2
        workplaces_sector = (self.width // 5, 3 * self.width // 5)
        shops_sector = (3 * self.width // 5, self.width)

        houses = self.create_buildings(
                houses_sector, "house", 1, self.houses_count)
        workplaces = self.create_buildings(
                workplaces_sector, "workplace", 3, self.workplaces_count)
        shops = self.create_buildings(
                shops_sector, "shop", 3, self.shops_count)
        self.buildings = houses + workplaces + shops
        for i, b in enumerate(self.buildings):
            b['id'] = i + 1

    def create_buildings(self, sector_dim, sector_type, margin_width, count):
        x = sector_dim[0] + margin_width
        y = margin_width
        building_width = (sector_dim[1] - sector_dim[0]) - 2 * margin_width
        building_height = (self.height - ((count + 1) * margin_width)) // count
        return [{
            # ids are assigned globaly later
            "bottom-left": (x, margin_width + (building_height + margin_width) * i),
            "width": building_width,
            "height": building_height,
            "type": sector_type}
            for i in range(count)]

    def parse_from_file(self, filepath):
        pass

default_config = Config()

class PandemicsModel(Model):
    def __init__(self, config=default_config, disease=dis.covid_disease):
        self.agents_count = config.citizens_count + config.policemen_count
        self.disease = disease
        self.deceased = []
        self.buried = []
        self.deceased_counter = 0
        self.infected_counter = 0
        self.grid = MultiGrid(config.width, config.height, True)
        self.safety_per_cell = np.ones((config.height, config.width))
        self.buildings_map = np.zeros((config.height, config.width))
        self.buildings_id_map = np.zeros((config.height, config.width))
        self.schedule = SimultaneousActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "deceased": "deceased_counter",
                "infected": "infected_counter"},
            agent_reporters={
                "hp": lambda a: a.profile["hp"],
                "mask_protection": "mask_protection",
                "infection_day": lambda a: a.profile["infection_day"],
                "obedience": lambda a: a.profile["obedience"],
                "fear": lambda a: a.profile["fear"]}
        )
        self.config = config
        self.buildings = {b["id"] : b for b in self.config.buildings}
        self.houses = [x for x in self.buildings.values() if x['type'] == 'house']
        self.workplaces = [x for x in self.buildings.values() if x['type'] == 'workplace']
        self.shops = [x for x in self.buildings.values() if x['type'] == 'shop']

        self.add_buildings_to_map(self.buildings)

        self.street_positions = []
        for x in range(self.config.width):
            for y in range(self.config.height):
                if self.buildings_map[y][x] == 0:
                    self.street_positions.append((x, y))

        self.house_to_agents = defaultdict(list)
        self.workplace_to_agents = defaultdict(list)

        self.current_location_type = None

        # Create agents
        for i in range(self.agents_count):
            if i < config.policemen_count:
                a = agent.create_distribution_policeman_agent(
                        i, self, config.policemen_mental_features_distribution)
                a.assign_house(self, self.houses)
            elif i < config.policemen_count + config.citizens_count:
                a = agent.create_distribution_citizen_agent(
                        i, self, config.citizens_mental_features_distribution)
                a.assign_house(self, self.houses)
                a.assign_workplace(self, self.workplaces)
            self.add_agent(a)

        for i in self.random.choices(self.schedule.agents, k=config.infected_count):
            i.start_infection()
        self.running = True
        self.steps_count = 0
        self.datacollector.collect(self)

    # Returns (type, id) of the building where agent a is currently located
    def where_is_agent(self, a):
        (x, y) = a.pos
        return (self.buildings_map[y][x], self.buildings_id_map[y][x])

    def compute_time_of_day(self):
        return self.steps_count % self.config.steps_per_day / (self.config.steps_per_day / HOURS_PER_DAY)

    def compute_current_location_type(self):
        t = self.compute_time_of_day()
        return self.config.day_plan[t] if t in self.config.day_plan else \
                self.config.day_plan[min(self.config.day_plan.keys(), key=lambda k: k-t)]

    # Updates current location type based on time of day and the config schedule
    # Returns true if there is a change in current_location_type
    def update_current_location_type(self):
        t = self.compute_time_of_day()
        if t in self.config.day_plan:
            self.current_location_type = self.config.day_plan[t]
            return True
        return False

    def add_buildings_to_map(self, buildings):
        for b in buildings.values():
            (x, y) = b["bottom-left"]
            for i in range(x, x+b["width"]):
                for j in range(y, y+b["height"]):
                    self.buildings_map[j][i] = self.config.building_tags[b['type']]
                    self.buildings_id_map[j][i] = b['id']

    def add_agent(self, a):
        self.schedule.add(a)
        # Add the agent to a random grid cell
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(a, (x, y))

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
        if (self.update_current_location_type()):
            for a in self.schedule.agents:
                if (self.current_location_type is not None):
                    b = a.select_building(self.current_location_type)
                    a.teleport_to_building(b)
                else:
                    a.teleport_to_street()

        self.evaluate_safety_per_cell()
        self.schedule.step()
        self.steps_count += 1
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

grid_width = 30
grid_height = 30
policemen_count = 10
citizens_count = 30
infected_initially_count = 5

test_rebels_policemen_config = Config(None, citizens_count, grid_width, grid_height)
test_rebels_policemen_config.policemen_count = policemen_count
test_rebels_policemen_config.citizens_mental_features_distribution["fear"] = (0, 0)
test_rebels_policemen_config.citizens_mental_features_distribution["obedience"] = (0, 0)
test_rebels_policemen_config.citizens_mental_features_distribution["awareness"] = (0.5, 0.2)

fig_height = grid_height * 0.4
fig_cols = 2
fig_rows = 2

model = PandemicsModel(test_rebels_policemen_config)
num_steps = 200

for i in range(num_steps):
    if (i % (model.config.steps_per_day / HOURS_PER_DAY) == 0):
        fig = plt.figure(figsize=(fig_height * fig_cols * grid_width/grid_height,fig_height))
        gs = gridspec.GridSpec(fig_rows, fig_cols, width_ratios=[1, 1], height_ratios=[10,1])
        ax0 = fig.add_subplot(gs[0, 0])
        cax0 = fig.add_subplot(gs[1, 0])
        vis.visualise_all_agents_position_and_covid_status(ax0, model)
        vis.visualise_buildings_map(ax0, cax0, model)
        ax1 = fig.add_subplot(gs[0, 1])
        cax1 = fig.add_subplot(gs[1, 1])
        #vis.heatmap_safety_per_cell(ax1, cax, model)
        vis.heatmap_all_agents_profile_feature(ax1, cax1, model, 'obedience')
        #display.clear_output(wait=True) # Uncomment to see plt imgs as animation
        display.display(plt.gcf())
        #time.sleep(0.1)
        plt.close(fig)

    model.step()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(
    fig_height * fig_cols * grid_width/grid_height,fig_height))
axes[0, 0].set_ylim([0, model.agents_count])
axes[0, 1].set_ylim([0, model.agents_count])
infected_count = model.datacollector.get_model_vars_dataframe()["infected"]
infected_count.plot(ax=axes[0, 0], title="infected")
deceased_count = model.datacollector.get_model_vars_dataframe()["deceased"]
deceased_count.plot(ax=axes[0, 1], title="deceased")
#vis.plot_all_agents_feature(axes[1, 0], model, "obedience")
#vis.plot_all_agents_feature(axes[1, 1], model, "infection_day")

for i in range(policemen_count, citizens_count):
    class A(object):
        pass
    a = A()
    a.unique_id = i
    vis.plot_agent_feature(axes[1,0], model, a, "obedience")
    vis.plot_agent_feature(axes[1,1], model, a, "mask_protection")

#for a in model.schedule.agents:
#    plot_agent_feature(a, 'hp')

#for a in model.buried:
#    plot_agent_feature(a, 'hp')


#    def create_random_buildings(self, minimum_margin=0):
#        self.houses_count = 5
#        self.workplaces_count = 5
#        self.shops_count = 5
#
#        building_types = ["house"] * self.houses_count + \
#                         ["workplace"] * self.workplaces_count + \
#                         ["shop"] * self.shops_count
#
#        def colides(a, b, minium_margin=0):
#            colides_on_x = (a[0] > b[0] - minimum_margin and a[0] < b[0] + b[2] + minimum_margin) or \
#            (b[0] > a[0] - minimum_margin and b[0] < a[0] + a[2] + minimum_margin)
#            colides_on_y = (a[1] > b[1] - minimum_margin and a[1] < b[1] + b[3] + minimum_margin) or \
#            (b[1] > a[1] - minimum_margin and b[1] < a[1] + a[3] + minimum_margin)
#            return colides_on_x and colides_on_y
#
#        buildings = []
#        for i in range(30000):
#            if len(buildings) >= 15:
#                break
#            # X, Y, Width, Height
#            building_candidate = (random.randrange(self.width), random.randrange(self.height),
#                    random.randrange(2, 5), random.randrange(2, 5))
#            if any([colides(building_candidate, b, minimum_margin) for b in buildings]):
#                continue
#            if (building_candidate[0] + building_candidate[2] > self.width):
#                continue
#            if (building_candidate[1] + building_candidate[3] > self.width):
#                continue
#            buildings.append(building_candidate)
#
#        random.shuffle(buildings)
#        ready_buildings = [{
#            "id": i+1,
#            "bottom-left": (b[0], b[1]),
#            "width": b[2],
#            "height": b[3],
#            "type": building_types[i]}
#            for i, b in enumerate(buildings)]
#
#        self.houses = ready_buildings[0: self.houses_count]
#        self.workplaces = ready_buildings[self.houses_count: self.houses_count+self.workplaces_count]
#        self.shops = ready_buildings[self.houses_count+self.workplaces_count:
#                self.houses_count+self.workplaces_count+self.shops_count]
