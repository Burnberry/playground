import math


def integrate(spell1, spell2):
    print('#'*5, "Integrating", '#'*5,)
    error = 0
    L = 0
    for i in range(len(spell1) - 1):
        x_11, y_11, x_12, y_12 = spell1[i] + spell1[i + 1]
        x_21, y_21, x_22, y_22 = spell2[i] + spell2[i + 1]


        e = (x_21 - x_11) ** 2 - (x_21 - x_11) * (x_22 - x_12) + (x_22 - x_12) ** 2 + (y_21 - y_11) ** 2 - (
                    y_21 - y_11) * (y_22 - y_12) + (y_22 - y_12) ** 2
        l = math.dist([x_21, y_21], [x_22, y_22])
        e *= l / 3

        print(e, l)

        L += l
        error += e
    print(error)
    print('#'*20)
    print()
    return error


def cross_spells(points):
    ver, hor = [], []

    for p in points:
        ver.append([0, p])
        hor.append([p, 0])

    return ver, hor


def point_spells(points):
    zero, one = [], []

    for p in points:
        zero.append([0, 0])
        one.append([p, 0])

    return zero, one
