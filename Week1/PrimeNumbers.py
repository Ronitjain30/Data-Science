list = [x for x in range(2,11) if all(x%i != 0 for i in range(2,x))]
print(list)