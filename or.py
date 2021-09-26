from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np


OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df