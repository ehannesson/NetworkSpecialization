# stats.py

import os


os.chdir('..')
from grow import Graph
os.chdir('statistics/')


def degree(G, nbunch=None):
    """
    Wrapper of networkx.degree. Returns a dictionary of {node : degree} pairs.
    If nbunch is omitted, then return degrees of *all* nodes.
    """
    return dict(G.nxG.degree)


def 
