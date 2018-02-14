import numpy as np

num = np.random.randn(5)
expa = np.exp(num)

ans = expa / expa.sum()   #probabilily (ans) adds up to 1


#100 rows of 5 cols each:
num = np.random.randn(100,5)

expa = np.exp(num)

ans = expa / expa.sum(axis=1, keepdims=True)

print(ans)