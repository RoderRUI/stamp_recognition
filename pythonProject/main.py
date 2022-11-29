# This is a sample Python script.
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(x)
a = np.nonzero(x)
print("-----")

b = np.random.randint(0, 200, (4000, 2000))
v = b.max()
print(v)
print(b)
# d = b.reshape(200, 200)
#
# e = np.arange(160)
# f = b.reshape(200, 200)

# g = d-f
# print(g)
g = 170 * np.ones((4000, 2000))
#
# f = 160 * np.ones((4000, 2000))
d = g - b
print()
print(d)
print(">>>>>>>>>>>>>")
num = d[d >= 0]
print(num)
print(">>>>>>>>>>>>>")
print(np.count_nonzero(num))
# b = np.busday_count(d)
# print(b)
# def print_hi():
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hello World')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
