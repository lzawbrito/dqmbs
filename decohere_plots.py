import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv("./data/decohere_2level.csv")
t = df['t'].to_numpy()
up_pop = df['up_pop'].to_numpy()
down_pop = df['down_pop'].to_numpy()
fig, ax = plt.subplots() 
# ax.plot(t, down_pop, color='lightgray')
ax.plot(t, up_pop, color='black')
# ax.set_xlim(0, 20)
plt.show()