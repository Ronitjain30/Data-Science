n = int(input())

def Fibonacci_generator():
    a = 0
    b = 1
    while True:
        yield a
        temp = a + b
        a = b
        b = temp
    
result = Fibonacci_generator()

for i in range(n):
    print(next(result))