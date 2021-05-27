class reverse_list:
    def __init__(self, list):
        self.list = list

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.list) == 0:
            raise StopIteration
        else:
            temp = self.list[len(self.list) - 1]
            self.list.pop()
            # print(len(self.list))
            return temp

r_list = reverse_list([2,3,4,5,6])

while(True):
    print(next(r_list))