import numpy as np
import math


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    Вычисляем моменты весовой функции с 0-го по max_s-ый на интервале [xl, xr]
    Весовая функция: p(x) = 1 / (x-a)^alpha / (b-x)^beta, причём гарантируется, что:
        1) 0 <= alpha < 1
        2) 0 <= beta < 1
        3) alpha * beta = 0

    :param max_s:   номер последнего момента
    :return:        список значений моментов
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        m = [((xr - a) ** (1 - alpha) - (xl - a) ** (1 - alpha)) / (1 - alpha)]
        for i in range(1, max_s + 1):
            coef = np.poly([-a] * i)[::-1]
            m.append(sum([coef[j] / (j + 1 - alpha) * ((xr - a) ** (j + 1 - alpha) - (xl - a) ** (j + 1 - alpha))
                          for j in range(0, i + 1)]))
        return m
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        m = [((b - xl) ** (1 - beta) - (b - xr) ** (1 - beta)) / (1 - beta)]
        for i in range(1, max_s + 1):
            coef = (-1) ** i * np.poly([b] * i)[::-1]
            m.append(- sum(
                [coef[j] / (j + 1 - beta) * ((b - xr) ** (j + 1 - beta) - (b - xl) ** (j + 1 - beta))
                 for j in range(0, i + 1)]))
        return m

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]


def runge(s0, s1, m, L):
    """
    Оценка погрешности последовательных приближений s0 и s1 по правилу Рунге

    :param m:   порядок погрешности
    :param L:   кратность шага
    :return:    оценки погрешностей s0 и s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оценка порядка главного члена погрешности по последовательным приближениям s0, s1 и s2 по правилу Эйткена
    Считаем, что погрешность равна R(h) = C*h^m + o(h^m)

    :param L:   кратность шага
    :return:    оценка порядка главного члена погрешности (m)
    """
    m = - np.log(np.abs((s2 - s1) / (s1 - s0))) / np.log(L)
    return m


def equidist(n: int, x0: float, x1: float):
    if n == 1:
        return [0.5 * (x0 + x1)]
    else:
        return np.linspace(x0, x1, n)


def quad(func, x0, x1, xs, **kwargs):
    """
    Интерполяционная квадратурная формула

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param xs:      узлы
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    n = len(xs)
    mat = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(mat), (n, n))
    a = np.linalg.solve(w, moments(n - 1, x0, x1, **kwargs))
    print(a)
    return sum(a * np.array([func(x) for x in xs]))
    m = moments(len(xs) - 1, x0, x1, **kwargs)


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    Интерполяционная квадратурная формула типа Гаусса

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param n:       количество узлов
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    mu = np.array(moments(2 * n - 1, x0, x1, **kwargs))
    w = np.array([[(mu[j + s]) for j in range(n)] for s in range(n)])
    b = - np.array(mu[n:])
    coef = np.append(np.linalg.solve(w, b), 1)
    xs = np.roots(coef[::-1])
    nodes = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(nodes), (n, n))
    a = np.linalg.solve(w, mu[:n])
    return sum(a * np.array([func(x) for x in xs]))


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    Составная квадратурная формула

    :param func:        интегрируемая функция
    :param x0, x1:      интервал
    :param n_intervals: количество интервалов
    :param n_nodes:     количество узлов на каждом интервале
    :param kwargs:      параметры весовой функции (должны передаваться в moments)
    :return:            значение СКФ
    """
    mesh = np.linspace(x0, x1, n_intervals + 1)
    return sum(quad(func, mesh[i], mesh[i + 1], equidist(n_nodes, mesh[i], mesh[i + 1]), **kwargs) for i in range(n_intervals))


def integrate(func, x0, x1, tol):
    """
    Интегрирование с заданной точностью (error <= tol)

    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге и выбираем оптимальный размер шага
    Делаем так, пока оценка погрешности не уложится в tol

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param tol:     допуск
    :return:        значение интеграла, оценка погрешности
    """
    n_nodes = 3
    L = 2
    h_opt = x1 - x0
    error = tol + 1
    while error >= tol:
        n_intervals = [math.ceil(int((x1 - x0) / h_opt)) * L ** s for s in range(3)]
        compos_h = [composite_quad(func, x0, x1, interval, n_nodes) for interval in n_intervals]
        m = aitken(*compos_h, L)
        error = runge(*compos_h[:-1], m, L)[0]
        h_opt = compos_h[0] * (tol / error) ** (1 / m)
        print(n_intervals)
    return compos_h[0], error
