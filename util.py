#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def elems_split(elems, background):
    elems_split = []
    for e in elems:
        elems_proc = [i for i in unicode(e).split('_') if i]

        if background:
            elems_proc = ['-background-'] + elems_proc

        elems_split.append(elems_proc)
    return elems_split


def make_dictionary(elems):
    dictionary = {}
    for elem in sorted(set(elems)):
        dictionary[elem] = len(dictionary)
    return dictionary


def reverse_dictionary(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))


def encode_elems(elems):
    elems_unique = set([c for cs in elems for c in cs])
    elems_dict = make_dictionary(elems_unique)
    enc_elems = []
    for row in elems:
        enc_elems.append([elems_dict[e] for e in row if e])
    return np.array(enc_elems), elems_dict


def build_dataset(cues, outcomes, freqs, background=False):
    cues_split = elems_split(cues, background=background)
    outcomes_split = elems_split(outcomes, background=False)
    cues, cues_dict = encode_elems(cues_split)
    outcomes, outcomes_dict = encode_elems(outcomes_split)
    probs = freqs/freqs.sum()
    return cues, cues_dict, outcomes, outcomes_dict, probs


def read_data(fname, add_background):
    data = pd.read_csv(fname, keep_default_na=False)

    cues, cues_dict, outcomes, outcomes_dict, probs = build_dataset(
        data['Cues'], data['Outcomes'], data['Frequency'],
        background=add_background)

    return cues, cues_dict, outcomes, outcomes_dict, probs
