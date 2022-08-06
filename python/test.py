import torch as t

a = t.arange(1, 9).reshape(2, 2, 2)

b = t.arange(3, 15).reshape(2, 2, 3)


print(a)
print(b)
print(t.tensordot(a, b, 2))