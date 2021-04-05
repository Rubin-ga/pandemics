#!/usr/bin/env python
# coding: utf-8

class Disease():
    def __init__(self, name, peak_day,
                 infectiousness_start,     # day when you get infectious
                 symptoms_start,           # day when you get symptomatic
                 duration,                 # duration
                 hp_change,                # hp change
                 mask_protection_factor):  # masks effectiveness against the virus
        self.name = name
        self.peak_day = peak_day
        self.infectiousness_start = infectiousness_start
        self.symptoms_start = symptoms_start
        self.duration = duration
        self.hp_change = lambda profile, disease=self: hp_change(disease, profile)
        self.mask_protection_factor = mask_protection_factor

def covid_hp_change(disease, profile):
    if (profile["infection_day"] < disease.peak_day):
        return -8
    if (profile["infection_day"] > disease.peak_day and
        profile["infection_day"] <= disease.duration):
        return 8
    return 0

covid_disease = Disease('covid',
                        peak_day = 5,
                        infectiousness_start = 3,
                        symptoms_start = 4,
                        duration = 10,
                        hp_change = covid_hp_change,
                        mask_protection_factor = 0.8)

