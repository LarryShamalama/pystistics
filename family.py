import numpy as np
import pandas as pd


class _family:
    '''
    latent class; should be not be accessed
    '''
    def __init__(self, link):
        assert link in ["logit", "identity", "inverse", "log"]

        self.link = link

class gaussian(_family):
    def __init__(self, link="identity"):
        super().__init__(link)