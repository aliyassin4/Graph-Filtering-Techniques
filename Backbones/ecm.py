import jax
import numpy as np
import jax.numpy as jnp
import warnings
import time
from abc import abstractmethod, ABC
from collections import namedtuple
import scipy.optimize
import scipy.sparse
from tabulate import tabulate
from jax import jit, jacfwd, jacrev, grad


#from .MaxentGraph import MaxentGraph
#from .util import EPS, R_to_zero_to_inf, R_to_zero_to_one, jax_class_jit

EPS = np.finfo(float).eps
R_to_zero_to_inf = [(jit(jnp.exp), jit(jnp.log)), (jit(jax.nn.softplus), softplus_inv)]
R_to_zero_to_one = [
    (jit(jax.nn.sigmoid), sigmoid_inv),
    (shift_scale_arctan, shift_scale_arctan_inv),
]

def hvp(f):
    """
    Hessian-vector-product

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode
    """
    return lambda x, v: jvp(grad(f), (x,), (v,))[1]

def wrap_with_array(f):
    # asarray gives fortran contiguous error with L-BFGS-B
    # have to use np.array
    # https://github.com/google/jax/issues/1510#issuecomment-542419523
    return lambda v: np.array(f(v))


def print_percentiles(v):
    """
    Prints the min, 25th percentile, median, 75th percentile, and max.
    """

    table = [
        ["Min", np.min(v)],
        ["25th", np.percentile(v, 25)],
        ["Median", np.median(v)],
        ["75th", np.percentile(v, 75)],
        ["Max", np.max(v)],
    ]
    table_str = tabulate(table, headers=["Percentile", "Relative error"])
    print(table_str)

def jax_class_jit(f):
    """
    Lets you JIT a class method with JAX.
    """
    return partial(jax.jit, static_argnums=(0,))(f)


### R <=> (0, inf) homeomorphisms
@jit
def softplus_inv(x):
    return jnp.log(jnp.exp(x) - 1)


"""
Contains ABC for Maximum Entropy graph null model.
"""

Solution = namedtuple(
    "Solution", ["x", "nll", "residual_error_norm", "relative_error", "total_time"]
)


class MaxentGraph(ABC):
    """
    ABC for Maximum Entropy graph null model.
    """

    @abstractmethod
    def bounds(self):
        """
        Returns the bounds on the parameters vector.
        """

    def clip(self, v):
        """
        Clips the parameters vector according to bounds.
        """
        (lower, upper), _bounds_object = self.bounds()
        return np.clip(v, lower, upper)

    @abstractmethod
    def transform_parameters(self, v):
        """
        Transforms parameters to bounded form.
        """

    @abstractmethod
    def transform_parameters_inv(self, v):
        """
        Transforms parameters to all real numbers for optimization convenience.
        """

    @abstractmethod
    def order_node_sequence(self):
        """
        Concatenates node constraint sequence in a canonical order.
        """

    @abstractmethod
    def get_initial_guess(self, option):
        """
        Gets initial guess.
        """

    @abstractmethod
    def expected_node_sequence(self, v):
        """
        Computes the expected node constraint using matrices.
        """

    @abstractmethod
    def expected_node_sequence_loops(self, v):
        """
        Computes the expected node constraint using loops.
        """

    @jax_class_jit
    def node_sequence_residuals(self, v):
        """
        Computes the residuals of the expected node constraint sequence minus the actual sequence.
        """
        return self.expected_node_sequence(v) - self.order_node_sequence()

    @abstractmethod
    def neg_log_likelihood_loops(self, v):
        """
        Computes the negative log-likelihood using loops.
        """

    @abstractmethod
    def neg_log_likelihood(self, v):
        """
        Computes the negative log-likelihood using matrix operations.
        """

    def compute_relative_error(self, expected):
        """
        Computes relative error for solution for every element of the sequence.
        """
        actual = self.order_node_sequence()

        # okay not actually relative error but close enough
        return np.abs(expected - actual) / (1 + np.abs(actual))

    def solve(self, x0, method="trust-krylov", verbose=False):
        """
        Solves for the parameters of the null model using either bounded minimization of the
        negative log-likelihood or bounded least-squares minimization of the equation residuals.
        """

        args = {}

        # for some reason scipy prefers hess over hessp if the former is passed
        # but since the latter is more efficient, only pass hess when necessary
        if method in ["trust-exact", "dogleg"]:
            hess = jit(jacfwd(jacrev(self.neg_log_likelihood)))
            args["hess"] = hess
        elif method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hessp = jit(hvp(self.neg_log_likelihood))
            args["hessp"] = hessp

        if method in ["trf", "dogbox", "lm"]:
            f = self.node_sequence_residuals
            jac = jit(jacrev(self.expected_node_sequence))
            args["jac"] = jac
            solver = scipy.optimize.least_squares
        elif method in [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]:
            f = self.neg_log_likelihood
            jac = jit(grad(self.neg_log_likelihood))

            # lbfgsb is fussy. wont accept jax's devicearray
            # there may be others, though
            if method in ["L-BFGS-B"]:
                jac = wrap_with_array(jac)

            if method in [
                "CG",
                "BFGS",
                "Newton-CG",
                "L-BFGS-B",
                "TNC",
                "SLSQP",
                "dogleg",
                "trust-ncg",
                "trust-krylov",
                "trust-exact",
                "trust-constr",
            ]:
                args["jac"] = jac

            solver = scipy.optimize.minimize
        else:
            raise ValueError("Invalid optimization method")

        start = time.time()
        sol = solver(f, x0=x0, method=method, **args)
        end = time.time()

        total_time = end - start

        eq_r = self.node_sequence_residuals(sol.x)
        expected = self.expected_node_sequence(sol.x)
        residual_error_norm = np.linalg.norm(eq_r, ord=2)
        relative_error = self.compute_relative_error(expected)
        nll = self.neg_log_likelihood(sol.x)

        if not sol.success:
            if np.max(relative_error) < 0.5:
                warnings.warn(
                    "Didn't succeed according to algorithm, but max relative error is low.",
                    RuntimeWarning,
                )
            else:
                raise RuntimeError(
                    f"Didn't succeed in minimization. Message: {sol.message}"
                )

        if verbose:
            print(f"Took {total_time} seconds")
            print("Relative error for expected degree/strength sequence: ")
            print()
            print_percentiles(relative_error)

            print(f"\nResidual error: {residual_error_norm}")

        return Solution(
            x=sol.x,
            nll=float(nll),
            residual_error_norm=residual_error_norm,
            relative_error=relative_error,
            total_time=total_time,
        )



class ECM(MaxentGraph):
    """
    (Undirected) Enhanced configuration model.
    """

    def __init__(self, W, x_transform=0, y_transform=0):
        # validate?

        # ignore self-loops
        W -= scipy.sparse.diags(W.diagonal())

        self.k = (W > 0).sum(axis=1).getA1().astype("float64")
        self.s = W.sum(axis=1).getA1()

        self.num_nodes = len(self.k)

        self.x_transform, self.x_inv_transform = R_to_zero_to_inf[x_transform]
        self.y_transform, self.y_inv_transform = R_to_zero_to_one[y_transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * 2 * self.num_nodes)
        upper_bounds = np.array([np.inf] * self.num_nodes + [1 - EPS] * self.num_nodes)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k, self.s])

    @jax_class_jit
    def transform_parameters(self, v):
        x = v[: self.num_nodes]
        y = v[self.num_nodes :]

        return jnp.concatenate((self.x_transform(x), self.y_transform(y)))

    @jax_class_jit
    def transform_parameters_inv(self, v):
        x = v[: self.num_nodes]
        y = v[self.num_nodes :]

        return jnp.concatenate((self.x_inv_transform(x), self.y_inv_transform(y)))

    def get_initial_guess(self, option=5):
        """
        Just some options for initial guesses.
        """
        num_nodes = len(self.k)
        num_edges = np.sum(self.k) / 2

        ks = self.k
        ss = self.s

        if option == 1:
            initial_guess = np.random.sample(2 * num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.01, 2 * num_nodes)
        elif option == 3:
            initial_guess = np.repeat(0.10, 2 * num_nodes)
        elif option == 4:
            initial_guess = np.concatenate([ks / ks.max(), ss / ss.max()])
        elif option == 5:
            initial_guess = np.concatenate(
                [ks / np.sqrt(num_edges), np.random.sample(num_nodes)]
            )
        elif option == 6:
            xs_guess = ks / np.sqrt(num_edges)
            s_per_k = ss / (ks + 1)
            ys_guess = s_per_k / s_per_k.max()
            initial_guess = np.concatenate([xs_guess, ys_guess])
        else:
            raise ValueError("Invalid option value. Choose from 1-6.")

        return self.transform_parameters_inv(self.clip(initial_guess))

    @jax_class_jit
    def expected_node_sequence(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        pij = xx * yy / (1 - yy + xx * yy)
        pij = pij - jnp.diag(jnp.diag(pij))
        avg_k = pij.sum(axis=1)

        sij = pij / (1 - yy)
        # don't need to zero out diagonal again, still 0
        avg_s = sij.sum(axis=1)

        return jnp.concatenate((avg_k, avg_s))

    def expected_node_sequence_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        avg_k = np.zeros(N)
        avg_s = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                xx = x[i] * x[j]
                yy = y[i] * y[j]

                pij = xx * yy / (1 - yy + xx * yy)

                avg_k[i] += pij
                avg_s[i] += pij / (1 - yy)

        return np.concatenate([avg_k, avg_s])

    def neg_log_likelihood_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = 0

        for i in range(N):
            llhood += self.k[i] * np.log(x[i])
            llhood += self.s[i] * np.log(y[i])

        for i in range(N):
            for j in range(i):
                xx = x[i] * x[j]
                yy = y[i] * y[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += np.log(t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = 0

        llhood += jnp.sum(self.k * jnp.log(x))
        llhood += jnp.sum(self.s * jnp.log(y))

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        t = (1 - yy) / (1 - yy + xx * yy)
        log_t = jnp.log(t)
        llhood += jnp.sum(log_t) - jnp.sum(jnp.tril(log_t))

        return -llhood

    def get_pval_matrix(self, v, W):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        # only need one triangle since symmetric
        # convert to lil for fast index assignment
        # convert to float because may be int at this point
        W_new = scipy.sparse.tril(W.copy()).tolil().astype(np.float64)

        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx_out = x[i] * x[j]
            yy_out = y[i] * y[j]
            pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)

            # probability this weight would be at least this large, given null model
            p_val = pij * np.power(y[i] * y[j], w - 1)
            W_new[i, j] = p_val

        return W_new
