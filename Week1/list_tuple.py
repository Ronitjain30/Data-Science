n = int(input("Enter the size of list: "))

l1 = []
for i in range(n):
    l1.append(int(input()))

l2 = [(x, l1[x]) for x in range(n)]

print(l2)