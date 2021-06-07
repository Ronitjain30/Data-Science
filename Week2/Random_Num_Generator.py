import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.randint(1, 101, 1000)    #Generating 1000 random numbers in range (1, 100)

s = pd.Series(x)

#To display full series uncomment this:-
# with pd.option_context('display.max_rows',None):
#     print(s.value_counts())

print(s.value_counts())                #Frequency Table

graph = s.plot.hist(bins=20, alpha=0.5)
plt.show()