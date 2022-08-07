import torch as t

# a = t.arange(1, 9).reshape(2, 2, 2)

# b = t.arange(3, 15).reshape(2, 2, 3)


# print(a)
# print(b)
# print(t.tensordot(a, b, 2))

a = t.arange(1.0, 5.0).reshape(2, 2)
b = t.arange(1.0, 5.0).reshape(2, 2)

a.requires_grad = True

c = a.matmul(b) * a

print("A", a)
print("B", b)
print("C", c)

c.sum().backward()

print(a.grad)
