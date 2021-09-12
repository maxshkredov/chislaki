import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic, HarmExp, Arenstorf
from utils.utils import get_accuracy
from solve_ode import adaptive_step_integration, AdaptType
from one_step_methods import ExplicitEulerMethod, RungeKuttaMethod, EmbeddedRungeKuttaMethod
import coeffs_collection as coeffs


@pytest.mark.parametrize('ode,y0', (
        (
                Harmonic(np.array([1., 1.]), 1, 1),
                np.array([1., 1.])
        ),
        (
                HarmExp(),
                np.exp([1, 0])
        ),
))
def test_adaptive(ode, y0):
    """
    Проверяем алгоритмы выбора шага
    """
    t0, t1 = 0, 4*np.pi

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )

    for method, adapt_type in methods:
        ode.clear_call_counter()
        ts, ys = adaptive_step_integration(method=method,
                                           ode=ode, y_start=y0, t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        print(f'{method.name} took {ode.get_call_counter()} function calls')

        tss.append(np.array(ts))
        yss.append(ys)

    ts = np.array(sorted([t for ts in tss for t in ts]))
    exact = ode[ts].T
    y0 = np.array([y[0] for y in exact])

    # plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.plot(ts, y0, 'ko-', label='exact')

    for (m, _), ts, ys in zip(methods, tss, yss):
        ax1.plot(ts, [y[0] for y in ys], '.', label=m.name)
        ax2.plot(ts[:-1], np.diff(ts), '.-', label=m.name)
        ax3.plot(ts, get_accuracy(ode[ts].T, ys), '.-', label=m.name)

    ax1.legend(), ax1.set_title('y(t)')
    ax2.legend(), ax2.set_title('dt(t)')
    ax3.legend(), ax3.set_title('accuracy')

    fig.suptitle('test_adaptive')
    fig.tight_layout()
    plt.show()


def test_adaptive_order():
    """
    Проверяем сходимость
    Q: почему наклон линии (число в скобках) соответствует порядку метода?
    """
    t0, t1 = 0, 2*np.pi
    y0 = np.array([1., 1.])
    ode = Harmonic(y0, 1, 1)

    methods = (
        (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.dopri_coeffs),         AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )
    tols = 10. ** -np.arange(3, 9)

    plt.figure(figsize=(9, 6))
    for i, (method, adapt_type) in enumerate(methods):
        print(method.name)
        fcs = []
        errs = []
        for tol in tols:
            ode.clear_call_counter()
            ts, ys = adaptive_step_integration(method=method,
                                               ode=ode,
                                               y_start=y0,
                                               t_span=(t0, t1),
                                               adapt_type=adapt_type,
                                               atol=tol, rtol=tol*1e3)
            err = np.linalg.norm(ys[-1] - ode[t1])
            fc = ode.get_call_counter()
            print(f'{method.name} with {adapt_type.name}: {fc} RHS calls, err = {err:.5f}')
            errs.append(err)
            fcs.append(fc)

        x = np.log10(fcs)
        y = -np.log10(errs)
        k, b = np.polyfit(x, y, 1)
        plt.plot(x, k*x+b, 'k:')
        plt.plot(x, y, 'p', label=f'{method.name} {adapt_type} ({k:.2f})')

    plt.title('test_adaptive_order: check RHS evals')
    plt.xlabel('log10(function_calls)')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()


def test_arenstorf():
    """
    https://en.wikipedia.org/wiki/Richard_Arenstorf#The_Arenstorf_Orbit
    https://commons.wikimedia.org/wiki/File:Arenstorf_Orbit.gif
    Q: какие участки траектории наиболее быстрые?
    """
    ode = Arenstorf()
    t0, t1 = 0, 1 * ode.t_period
    y0 = ode.y0

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        # (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    for method, adapt_type in methods:
        ts, ys = adaptive_step_integration(method=method,
                                           ode=ode,
                                           y_start=y0,
                                           t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        tss.append(np.array(ts))
        yss.append(ys)

    for (m, _), ts, ys in zip(methods, tss, yss):
        ax1.plot([y[0] for y in ys],
                 [y[1] for y in ys],
                 ':', label=m.name)
        ax2.plot(ts[:-1], np.diff(ts), '.-', label=m.name)

    derivatives = [np.log10(np.linalg.norm(ode(t, y))) for t, y in zip(ts, ys)]
    # derivatives = [1/(np.linalg.norm(ode(t, y))) for t, y in zip(ts, ys)]

    ax1.plot(0, 0, 'bo', label='Earth')
    ax1.plot(1, 0, '.', color='grey', label='Moon')
    ax3.plot(ts, derivatives, label='log10(|f(t, y)|)')

    ax1.legend(), ax1.set_title('trajectory')
    ax2.legend(), ax2.set_title('dt(t)')
    ax3.legend(), ax3.set_title('right-hand side')

    fig.suptitle('Arenstorf orbit')
    fig.tight_layout()

    plt.show()
