import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import VanDerPol
from solve_ode import adaptive_step_integration, AdaptType
from one_step_methods import (
    ExplicitEulerMethod,
    ImplicitEulerMethod,
    EmbeddedRosenbrockMethod,
)
import coeffs_collection as coeffs


def test_stiff():
    """
    Проверяем явные и неявные методы на жёсткой задаче
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    Q: почему даже метод Розенброка иногда уменьшает шаг почти до нуля?
    """
    t0 = 0
    t1 = 2500

    mu = 1000
    y0 = np.array([2., 0.])
    ode = VanDerPol(y0, mu)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    colors = 'rgbcmyk'
    for i, (method, adapt_type) in enumerate(
            [
                (ExplicitEulerMethod(),                                AdaptType.RUNGE),
                (ImplicitEulerMethod(),                                AdaptType.RUNGE),
                (EmbeddedRosenbrockMethod(coeffs.rosenbrock23_coeffs), AdaptType.EMBEDDED),
            ]
    ):
        ode.clear_call_counter()
        ts, ys = adaptive_step_integration(method, ode, y0, (t0, t1),
                                           adapt_type=adapt_type,
                                           atol=1e-6, rtol=1e-3)
        print(f'{method.name}: {len(ts)-1} steps, {ode.get_call_counter()} RHS calls')

        axs[0, 0].plot([y[0] for y in ys], [y[1] for y in ys], f'{colors[i]}.--', label=method.name)
        axs[0, 1].plot(ts[:-1], np.diff(ts), f'{colors[i]}.--', label=method.name)
        axs[1, 0].plot(ts, [y[0] for y in ys], f'{colors[i]}.--', label=method.name)
        axs[1, 1].plot(ts, [y[1] for y in ys], f'{colors[i]}.--', label=method.name)

    axs[0, 0].legend(), axs[0, 0].set_title('y(x)')
    axs[0, 1].legend(), axs[0, 1].set_title('dt(t)')
    axs[1, 0].legend(), axs[1, 0].set_title('x(t)')
    axs[1, 1].legend(), axs[1, 1].set_title('y(t)')

    fig.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}')
    fig.tight_layout()
    plt.show()
