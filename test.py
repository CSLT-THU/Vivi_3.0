import numpy as np
x = np.array(['z','p','z'])
y = np.array(['z','z','z'])
print(np.where(x==y,1.0,0.0))
print(np.where(x==y,float(1),float(0)))

pz = 'p'
li = [pz] * 44
print(type(li))
print(li)

print(np.array([np.zeros(4777)]*80))
print(type(np.array([np.zeros(4777)]*80)))