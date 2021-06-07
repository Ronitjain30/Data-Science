import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.randint(1, 101, 1000)                    #Generating 1000 random numbers in range (1, 100)
s = pd.Series(x)

s[~s.isin(s.value_counts().index[:5])] = 'Other'       #Retaining top 5 frequencies and ranaming others as 'Other'

freq_table = s.value_counts()                          #Frequency Table
print(freq_table)

graph = freq_table.plot.bar(x='Data', y='Frequency')
plt.show()