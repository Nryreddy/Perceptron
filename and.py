from utils.model import Perceptron
from utils.all__utils import save_model
from utils.all__utils import save_plot
from utils.all__utils import prepare_data
import pandas as pd
import numpy as np

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)

print(df)

X,y = prepare_data(df)

ETA = 0.3
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y) # training

_ = model.total_loss()

save_model(model,"and.model")
save_plot(df,"and.png",model)
