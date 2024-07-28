import math
import numpy as np


def spell_test():
    # print("### True error (no scaling/translating)")
    # print("The same figures should have the same error independent of scale and number of section\n")
    # print("# slope")
    # integrate_true(*slope_spells([0, 1]))
    # integrate_true(*slope_spells([0, 0.2, 1]))
    # integrate_true(*slope_spells([0, 0.2, 2]))
    # print()
    # print("# Cross")
    # integrate_true(*cross_spells([-0.5, 0, 0.5]))
    # integrate_true(*cross_spells([-1, 1]))
    # integrate_true(*cross_spells([-1, 0.2, 1]))
    # print()

    print("### Optimized error (optimize by scaling/translating)")
    print()
    P, Q = transformed_spells([-1, 0,  1], a=4, dx=5, dy=-10)
    # transform(Q, 1, 0, 1)
    # print(optimize(P, Q))
    # integrate_true(P, Q)
    P, Q = slope_spells([0, 1])
    transform(Q, 1, 0, 0)
    print(compute_error(P, Q))
    P, Q = slope_spells([0, 0.2, 1])
    transform(Q, 2, 5, -3)
    print(compute_error(P, Q))
    print("# square test")
    l = 10
    P = [[l, l], [-l, l], [-l, -l], [l, -l], [l, l]]
    Q = [[11, 11], [-9, 9], [-8, -11], [11, -10], [11, 11]]
    Q2 = [[0, 5], [3, 8], [10, -10], [10, 10], [-10, 10]]
    integrate_true(P, Q)
    print(compute_error(P, Q))
    integrate_true(P, Q2)
    print(compute_error(P, Q2))
    print("# letter tests")
    C = letter_c(1)
    Z = letter_z(10)
    print(compute_error(C, Z))
    print(compute_error(Z, C))


def compute_error(P, Q):
    a, dx, dy = optimize(P, Q)
    # todo negative a -> discard
    # point mirror not allowed
    # also a = 0? May be discarding
    error = integrate_true(P, Q)
    return error, error and 1/error or -1

def optimize(P, Q):
    """
    :param P list of points:
    :param Q: list of points
    :return:
    """
    A2, A = 0, 0
    DX2, DX = 0, 0
    DY2, DY = 0, 0
    ADX, ADY = 0, 0

    for P1, P2, Q1, Q2 in zip(P[:-1], P[1:], Q[:-1], Q[1:]):
        l = math.dist(P1, P2)

        a2, dx2, adx, a, dx = get_variable_terms(P1[0], P2[0], Q1[0], Q2[0])
        A2 += a2*l
        DX2 += dx2*l
        ADX += adx*l
        A += a*l
        DX += dx*l

        a2, dy2, ady, a, dy = get_variable_terms(P1[1], P2[1], Q1[1], Q2[1])
        A2 += a2*l
        DY2 += dy2*l
        ADY += ady*l
        A += a*l
        DY += dy*l

    # print('paras', A2, A, DX2, DX, DY2, DY, ADX, ADY)

    # c1*A + c2*DX + c3*DY + c4 = 0
    # [c1, c2, c3, c4]
    eq_a = [2*A2, ADX, ADY, A]
    eq_dx = [ADX, 2*DX2, 0, DX]
    eq_dy = [ADY, 0, 2*DY2, DY]

    eqs = np.array([eq_a[:-1], eq_dx[:-1], eq_dy[:-1]])
    consts = -np.array([eq_a[-1], eq_dx[-1], eq_dy[-1]])
    sol = np.linalg.solve(eqs, consts)
    # print('sol', sol)

    a, dx, dy = (float(i) for i in list(sol))
    transform(Q, a, -dx, -dy)  # messed up somewhere? translation needs to be done in opposite direction

    return a, dx, dy


def get_variable_terms(x1, x2, y1, y2):
    a2 = y1 ** 2 + y2 * y1 + y2 ** 2
    d2 = 3
    ad = -3 * (y1 + y2)
    a = -x1 * (2 * y1 + y2) - x2 * (y1 + 2 * y2)
    d = 3 * (x1 + x2)
    return a2, d2, ad, a, d


def transform(P, a, dx, dy):
    for i in range(len(P)):
        x, y = P[i]
        P[i] = [a*x + dx, a*y + dy]


def integrate_true(P, Q):
    E, L = 0, 0
    for P1, P2, Q1, Q2 in zip(P[:-1], P[1:], Q[:-1], Q[1:]):
        e = 0
        for i in range(len(P1)):
            e += integrate_section(P1[i], P2[i], Q1[i], Q2[i])
        l = math.dist(P1, P2)
        # print(e, l, '|', e*l)
        e *= l
        L += l
        E += e
    # print('E', E, 'L', L, '->', E/L**3)
    return E/L**3


def integrate_section(x1, x2, y1, y2):
    A = y1-x1
    B = y2-x2
    # print(x1, x2, y1, y2, A, B)
    return A**2 + A*B + B**2


def slope_spells(points):
    zero, one = [], []

    for p in points:
        zero.append([p, 0])
        one.append([p, p])

    return zero, one


def cross_spells(points):
    ver, hor = [], []

    for p in points:
        ver.append([0, p])
        hor.append([p, 0])

    return ver, hor


def transformed_spells(points, a=1, dx=0, dy=0):
    line1, line2 = [], []

    for p in points:
        line1.append([p, 0])
        line2.append([a*p+dx, dy])

    return line1, line2


def letter_c(l):
    return [[0, 0], [-l, 0], [-l, -2*l], [0, -2*l]]


def letter_z(l):
    return [[0, 0], [l, 0], [0, -2*l], [l, -2*l]]
