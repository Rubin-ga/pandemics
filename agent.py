#!/usr/bin/env python
# coding: utf-8

from mesa import Agent, Model
import numpy as np
import copy

""" Agent Mental Profile """
""" Obedience
0 - disobedience, 1 - perfect obedience;
inc: tickets;
dec: rumours;
"""
""" Fear
0 - no fear, 1 - intense fear;
inc: many COVID cases; severe COVID cases;
dec: less COVID cases, mild COVID cases
"""
""" Awareness
0 - ignorance, 1 - full awareness;
inc: tickets, COVID cases;
dec: rumours;
"""
physical_features = ['age', 'hp', 'deceased', 'infection_day']
mental_features = ['fear', 'awareness', 'obedience']

physical_features = ['age', 'hp', 'deceased', 'infection_day']
mental_features = ['fear', 'awareness', 'obedience']

def create_physical_profile_random(model):
    profile = dict()
    profile['age'] = model.random.randrange(0, 100)
    profile['hp'] = model.random.randrange(30, 100)
    profile['infection_day'] = 0
    profile['deceased'] = False
    return profile

def create_mental_profile_random(model):
    profile = dict()
    profile['obedience'] = model.random.random()
    profile['fear'] = model.random.random()
    profile['awareness'] = model.random.random()
    return profile

def create_mental_profile_distribution(model, distribution):
    profile = dict()
    for feature, dist in distribution.items():
        profile[feature] = np.random.normal(dist[0], dist[1])
    return profile

def create_mental_profile_perfect_citizen(model):
    profile = dict()
    profile['obedience'] = 1.0
    profile['fear'] = 1.0
    profile['awareness'] = 1.0

def create_mental_profile_rebel_citizen(model):
    profile = dict()
    profile['obedience'] = 0
    profile['fear'] = 0
    profile['awareness'] = 0

def create_random_citizen_agent(unique_id, model):
    return CitizenAgent(unique_id, create_physical_profile_random(model),
                        create_mental_profile_random(model), model)

def create_distribution_citizen_agent(unique_id, model,
        mental_features_distribution, physical_features_distribution=None):
    return CitizenAgent(unique_id,
            create_physical_profile_random(model),
            create_mental_profile_distribution(model, mental_features_distribution),
            model)

def create_distribution_policeman_agent(unique_id, model,
        mental_features_distribution, physical_features_distribution=None):
    return PolicemanAgent(unique_id,
            create_physical_profile_random(model),
            create_mental_profile_distribution(model, mental_features_distribution),
            model)

def create_random_policeman_agent(unique_id, model):
    return PolicemanAgent(unique_id, create_physical_profile_random(model),
                        create_mental_profile_random(model), model)
class CitizenAgent(Agent):
    """ An agent of a specified profile that describes agent's health and attitude"""

    def __init__(self, unique_id, physical_profile, mental_profile, model):
        super().__init__(unique_id, model)

        self.prev_pos = None
        self.next_pos = None

        self.profile = dict(physical_profile, **mental_profile)
        self.next_profile = self.profile.copy()
        """ 0 - no mask, 0.5 - cotton mask, 1 - KN95 mask """
        self.mask_protection = self.calculate_mask_protection(self.profile)
        self.next_mask_protection = self.calculate_mask_protection(self.next_profile)
        """ 0 - do not avoid, 0.5 - avoid unmasked, 1 - avoid all """
        self.avoid_others = self.calculate_avoid_others(self.profile)
        self.next_avoid_others = self.calculate_avoid_others(self.next_profile)

        self.modifiable_mental_features = mental_features

    def assign_house(self, model, houses):
        house = model.random.choice(houses)
        model.house_to_agents[house["id"]].append(self.unique_id)

    def assign_workplace(self, model, workplaces):
        workplace = model.random.choice(workplaces)
        model.workplace_to_agents[workplace["id"]].append(self.unique_id)

    def start_infection(self):
        self.profile["infection_day"] = 1
        self.model.infected_counter += 1

    def plan_start_infection(self):
        self.next_profile["infection_day"] = 1
        self.model.infected_counter += 1

    def is_infected(self):
        return self.profile["infection_day"] > 0 and not self.is_immune()

    def is_infectious(self):
        return self.profile["infection_day"] >= self.model.disease.infectiousness_start and not self.is_immune()

    def is_symptomatic(self):
        return self.profile["infection_day"] >= self.model.disease.symptoms_start and not self.is_immune()

    def is_immune(self):
        return self.profile["infection_day"] > self.model.disease.duration

    def prepare_for_ticket(self):
        self.next_profile["obedience"] += self.model.config.ticket_impact
        self.next_profile["obedience"] = min(1.0, self.next_profile["obedience"])

    def plan_individual_actions(self, cellmates):
        pass

    def plan_pass_away(self):
        self.next_profile["deceased"] = True
        self.model.deceased.append(self)

    def plan_move(self, possible_steps):
        max_safety = 0
        most_safe_cell = self.pos
        tolerable_possible_steps = []
        for ps in possible_steps:
            s = self.model.safety_per_cell[ps[1]][ps[0]]
            if s >= self.avoid_others:
                tolerable_possible_steps.append(ps)
            if s > max_safety:
                most_safe_cell = ps
        new_position = self.pos
        if len(tolerable_possible_steps) > 0:
            new_position = self.random.choice(tolerable_possible_steps)
        else:
            new_position = most_safe_cell
        self.next_pos = new_position

    def move(self):
        self.prev_pos = self.pos
        self.model.grid.move_agent(self, self.next_pos)

    def plan_set_protection(self):
        self.next_mask_protection = self.calculate_mask_protection(self.next_profile)
        self.next_avoid_others = self.calculate_avoid_others(self.next_profile)

    def calculate_mask_protection(self, profile):
        return profile["awareness"] * max(profile["fear"], profile["obedience"])

    def calculate_avoid_others(self, profile):
        return profile["awareness"] * max(profile["fear"], profile["obedience"])

    def set_protection(self):
        self.mask_protection = self.next_mask_protection
        self.avoid_others = self.next_avoid_others

    def plan_update_physical_profile(self, cellmates):
        if self.is_infected():
            if self.model.steps_count % self.model.config.steps_per_day == 0:
                self.next_profile["infection_day"] += 1
                self.next_profile["hp"] += self.model.disease.hp_change(self.profile)
                if self.next_profile["hp"] <= 0:
                    self.plan_pass_away()
        elif not self.is_immune(): # Susceptible to infection
            for c in filter(lambda c: c.is_infectious(), cellmates):
                risk = self.model.random.random()
                if risk > self.mask_protection * c.mask_protection \
                        * self.model.disease.mask_protection_factor :
                    self.plan_start_infection()
                    break

    def plan_update_mental_profile(self, cellmates):
        # cellmates is not empty: it includes the agent itself
        for m in self.modifiable_mental_features:
            avg_feature = sum([c.profile[m] for c in cellmates])/len(cellmates)
            if avg_feature > self.profile[m]:
                self.next_profile[m] += min(avg_feature - self.profile[m],
                                           self.model.config.group_pressure_inc)
                self.next_profile[m] = min(1.0, self.next_profile[m])
            if avg_feature < self.profile[m]:
                self.next_profile[m] += max(avg_feature - self.profile[m],
                                           -self.model.config.group_pressure_inc)
                self.next_profile[m] = max(0.0, self.next_profile[m])

    def update_profile(self):
        self.profile = self.next_profile.copy()

    def step(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        cellmates = self.model.grid.get_cell_list_contents(possible_steps)
        self.plan_update_physical_profile(cellmates)
        if self.next_profile["deceased"] == True:
            return
        self.plan_update_mental_profile(cellmates)
        self.plan_set_protection()
        self.plan_move(possible_steps)
        self.plan_individual_actions(cellmates)

    def advance(self):
        self.update_profile()
        if self.profile["deceased"] == True:
            return
        self.set_protection()
        self.move()

class PolicemanAgent(CitizenAgent):
    """ A fearless and perfectly obedient agent """
    def __init__(self, unique_id, physical_profile, mental_profile, model):
        mental_profile['obedience'] = 1.0
        mental_profile['fear'] = 0
        super().__init__(unique_id, physical_profile, mental_profile, model)
        self.modifiable_mental_features = ['awareness']

    def plan_individual_actions(self, cellmates):
        for c in filter(lambda c: c.mask_protection < 0.5, cellmates):
            c.prepare_for_ticket()
