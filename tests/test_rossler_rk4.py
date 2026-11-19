"""The shared Rossler RK4 step equals the expression that the engines used inline."""

import jax
import jax.numpy as jnp
import numpy as np

import fl_glider


def frozen_inline_rk4(x, y, z, c, P, fx, fy, dt):
    """Frozen copy of the block that every engine carried before the extraction.

    Only the layout differs from the original: the lambda is a def and the formatter
    set the spacing. Do not edit the expressions: this copy is the reference for the
    published numbers of Parts 5-19.
    """

    def f(x, y, z):
        return (-y - z + fx, x + P["a"] * y + fy, P["b"] + z * (x - c))

    k1 = f(x, y, z)
    k2 = f(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
    k3 = f(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
    k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
    xn = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    yn = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    zn = jnp.clip(z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), -50.0, 80.0)
    return xn, yn, zn


def shared_rk4(x, y, z, c, P, fx, fy, dt):
    """Call the shared function the way the engines call it."""
    return fl_glider.rossler_rk4_step(
        x, y, z, c=c, a=P["a"], b=P["b"], forcing_x=fx, forcing_y=fy, dt=dt
    )


def random_inputs(seed: int) -> tuple:
    """Return float32 oscillator fields; z spans both clip bounds, -50 and 80."""
    rng = np.random.default_rng(seed)
    shape = (12, 12)

    def field(low: float, high: float) -> jax.Array:
        return jnp.asarray(rng.uniform(low, high, shape).astype(np.float32))

    x, y, z, c = field(-8.0, 8.0), field(-8.0, 8.0), field(-70.0, 120.0), field(2.5, 5.0)
    forcing_x, forcing_y = field(-0.5, 0.5), field(-0.5, 0.5)
    parameters = fl_glider.params({})
    return x, y, z, c, parameters, forcing_x, forcing_y, parameters["dt"]


def test_shared_rk4_equals_frozen_inline_copy() -> None:
    """Goal: the extraction changed no bit. Method: exact equality on random inputs.

    The comparison runs eagerly and under jax.jit, the way the engines run it.
    """
    for seed in range(5):
        inputs = random_inputs(seed)
        for wrap in (lambda function: function, jax.jit):
            expected = wrap(frozen_inline_rk4)(*inputs)
            actual = wrap(shared_rk4)(*inputs)
            for expected_field, actual_field in zip(expected, actual):
                assert actual_field.dtype == jnp.float32
                assert np.array_equal(np.asarray(expected_field), np.asarray(actual_field))
        z_next = np.asarray(actual[2])
        assert z_next.min() == -50.0
        assert z_next.max() == 80.0
