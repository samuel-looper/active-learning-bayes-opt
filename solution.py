import random
import os
import typing
import logging
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt

EXTENDED_EVALUATION = False
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
ACQUISITION_FUNCTIONS = [
    "gp_lower_confidence",
    "constraint_weighted_ei",
]  # list of acquisition functions which have been implemented

""" Solution """


class BO_algo(object):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        # ACQUISITION FUNCTION PARAMETERS
        self.kappa = 1  # hyperparameter for GP lower confidence bound
        self.prev_best_z = np.inf  # previous 'best' sample
        self.acquisition_function_type = ACQUISITION_FUNCTIONS[1]

        print(f"-- Using acquisition function: '{self.acquisition_function_type}")

        self.previous_points = []  # previous evaluation points

        # IMPORTANT: DO NOT REMOVE THOSE ATTRIBUTES AND USE sklearn.gaussian_process.GaussianProcessRegressor instances!
        # Otherwise, the extended evaluation will break.

        # following project outline for kernel parameters
        self.constraint_model = GaussianProcessRegressor(
            kernel=RBF(length_scale=2, length_scale_bounds="fixed")
            * ConstantKernel(constant_value=3.5, constant_value_bounds="fixed")
        )
        self.objective_model = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.5, length_scale_bounds="fixed")
            * ConstantKernel(constant_value=1.5, constant_value_bounds="fixed")
        )

    def next_recommendation(self) -> np.ndarray:
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # recommend x which maximizes the acquisition function
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self) -> np.ndarray:  # DON'T MODIFY THIS FUNCTION
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that approximately maximizes the acquisition function.
        """

        def objective(x: np.array):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * np.random.rand(1)
            result = fmin_l_bfgs_b(
                objective, x0=np.array([x0, x1]), bounds=domain_x, approx_grad=True
            )
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f

        Returns
        ------
        af_value: float
            value of the acquisition function at x
        """

        if self.acquisition_function_type == "constraint_weighted_ei":
            """Return the constraint weighted expected improvement function

            For an overview please refer to Gelbart et al. (2014)
            """
            # evaluate the constraint and objective GPs at X
            obj_mu, obj_sigma = self.objective_model.predict(
                x.reshape(1, -1), return_std=True
            )
            constr_mu, constr_sigma = self.constraint_model.predict(
                x.reshape(1, -1), return_std=True
            )

            # probabilitiy that the constraint is <= 0 is the cdf up to 0
            pr_constr = norm.cdf(0, loc=constr_mu, scale=constr_sigma)

            # define eval_point = (prev_best_z - mu)/sigma, as per Gelbart et al. (2014)
            if self.prev_best_z == np.inf:
                # if there is no previous best, assume zero (equivalently: prev_best_z == obj_mu)
                eval_point = 0
            else:
                eval_point = (self.prev_best_z - obj_mu) / obj_sigma

            # compute expected improvement
            n = norm(loc=0, scale=1)
            ei_obj = obj_sigma * (eval_point * n.cdf(eval_point) + n.pdf(eval_point))

            # multiply EI by the probability of the constraint being met
            return ei_obj * pr_constr

        elif self.acquisition_function_type == "gp_lower_confidence":
            """Return the negative lower confidence bound

            Note:
            - The negative is returned so that the acquisition function can be maximized
            - This method does not take the constraint into account
            """
            (
                mu,
                sigma,
            ) = self.objective_model.predict(x.reshape(1, -1), return_std=True)
            return -(mu - self.kappa * sigma)

        else:
            raise NotImplementedError(
                f"Acquisition function type '{self.acquisition_function_type}' has not been implemented"
            )

    def add_data_point(self, x: np.ndarray, z: float, c: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        z: np.ndarray
            value of the acquisition function at x
        c: np.ndarray
            value of the condition function at x
        """

        assert x.shape == (1, 2)

        curr_point = np.array([[float(x[:, 0]), float(x[:, 1]), float(z), float(c)]])

        if len(self.previous_points) == 0:
            self.previous_points = curr_point
        else:
            # could cast a list to np array every iteration, but just take the
            # hit of using vstack
            self.previous_points = np.vstack((self.previous_points, curr_point))

        # track the current best point which ALSO meets the constraint
        if float(z) < self.prev_best_z and float(c) < 0:
            self.prev_best_z = float(z)

        # fit the Gaussian process prior
        self.constraint_model.fit(
            self.previous_points[:, 0:2], self.previous_points[:, 3]
        )
        self.objective_model.fit(
            self.previous_points[:, 0:2], self.previous_points[:, 2]
        )

    def get_solution(self) -> np.ndarray:
        """
        Return x_opt that is believed to be the minimizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # find the minimum values which also meets the constraint
        idx_with_valid_constr = np.where(self.previous_points[:, 3] < 0)[0]

        if len(idx_with_valid_constr) > 0:
            # something which meets the constraint exists
            min_idx_constr = idx_with_valid_constr[
                np.argmin(self.previous_points[idx_with_valid_constr, 2])
            ]
            x0_min, x1_min, z_min, constr_min = self.previous_points[min_idx_constr]

        else:
            # nothing meets the constraint; just return the minimum
            print(
                "-- WARNING: could not find any sample points which met the constraint"
            )
            min_idx_unconstr = np.argmin(self.previous_points[:, 2])
            x0_min, x1_min, z_min, constr_min = self.previous_points[min_idx_unconstr]

        x_min = np.array([[x0_min, x1_min]])

        print(f"-- Found min z: {z_min}")
        print(f"-- Found x value: {x_min}")
        print(f"-- Has constraint: {constr_min}")

        return x_min


""" 
    Toy problem to check  you code works as expected
    IMPORTANT: This example is never used and has nothing in common with the task you
    are evaluated on, it's here only for development and illustration purposes.
"""
domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = (
    -0.8
)  # This is an offset you can change to make the constraint more or less difficult to fulfill
LAMBDA = 0.0  # You shouldn't change this value


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(samples)
    true_cond = c(samples)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][
        np.argmin(true_values[np.where(true_cond < LAMBDA)])
    ]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))
    return f_opt, f_max, x_opt, x_max


def perform_extended_evaluation(agent, output_dir="./"):
    fig = plt.figure(figsize=(25, 5), dpi=50)
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)
    x_b, y_b = agent.get_solution().reshape(
        -1,
    )  # reshape to unpack values properly
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T
    predictions, stds = agent.objective_model.predict(samples, return_std=True)
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    conds = agent.constraint_model.predict(samples)
    conds = np.reshape(conds, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    true_values = f(samples)
    true_cond = c(samples)
    conditions_verif = (true_cond < LAMBDA).astype(float)
    conditions_with_nans = 1 - np.copy(conditions_verif)
    conditions_with_nans[np.where(conditions_with_nans == 0)] = np.nan
    conditions_with_nans = np.reshape(
        conditions_with_nans, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    valid_data_idx = np.where(true_cond < LAMBDA)[0]

    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][
        np.argmin(true_values[np.where(true_cond < LAMBDA)])
    ]

    sampled_point = np.array(agent.previous_points)

    ax_condition = fig.add_subplot(1, 4, 4)
    im_cond = ax_condition.pcolormesh(
        xv,
        yv,
        conds.reshape((EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)),
        shading="auto",
        linewidth=0,
    )
    im_cond.set_edgecolor("face")
    fig.colorbar(im_cond, ax=ax_condition)
    ax_condition.scatter(
        sampled_point[:, 0],
        sampled_point[:, 1],
        cmap="Blues",
        marker="x",
        label="Sampled Point by BO",
        antialiased=True,
        linewidth=0,
    )
    ax_condition.pcolormesh(
        xv,
        yv,
        conditions_with_nans,
        shading="auto",
        cmap="Reds",
        alpha=0.7,
        vmin=0,
        vmax=1.0,
        linewidth=0,
        antialiased=True,
    )
    ax_condition.set_title(
        "Constraint GP Posterior +  True Constraint (Red is Infeasible)"
    )
    ax_condition.legend(fontsize="x-small")

    ax_gp_f = fig.add_subplot(1, 4, 2, projection="3d")
    ax_gp_f.plot_surface(
        X=xv,
        Y=yv,
        Z=predictions,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False,
    )
    ax_gp_f.set_title("Posterior 3D for Objective")

    ax_gp_c = fig.add_subplot(1, 4, 3, projection="3d")
    ax_gp_c.plot_surface(
        X=xv, Y=yv, Z=conds, rcount=100, ccount=100, linewidth=0, antialiased=False
    )
    ax_gp_c.set_title("Posterior 3D for Constraint")

    ax_predictions = fig.add_subplot(1, 4, 1)
    im_predictions = ax_predictions.pcolormesh(
        xv,
        yv,
        predictions,
        shading="auto",
        label="Posterior",
        linewidth=0,
        antialiased=True,
    )
    im_predictions.set_edgecolor("face")
    fig.colorbar(im_predictions, ax=ax_predictions)
    ax_predictions.pcolormesh(
        xv,
        yv,
        conditions_with_nans,
        shading="auto",
        cmap="Reds",
        alpha=0.7,
        vmin=0,
        vmax=1.0,
        label=" True Infeasible",
        linewidth=0,
        antialiased=True,
    )
    ax_predictions.scatter(x_b, y_b, s=20, marker="x", label="Predicted Value by BO")
    ax_predictions.scatter(
        x_opt[0], x_opt[1], s=20, marker="o", label="True Optimimum Under Constraint"
    )
    ax_predictions.set_title(
        "Objective GP Posterior + True Constraint (Red is Infeasible)"
    )
    ax_predictions.legend(fontsize="x-small")
    figure_path = os.path.join(output_dir, "extended_evaluation.pdf")
    fig.savefig(figure_path)
    plt.show()


def train_on_toy(agent, iteration):
    logging.info("Running model on toy example.")
    seed = 1234
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {domain_x.shape[0]})"
        )

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0]), (
        f"The function get solution must return a numpy array of shape ("
        f"1, {domain_x.shape[0]})"
    )

    assert check_in_domain(solution), (
        f"The function get solution must return a point within the "
        f"domain, {solution} returned instead"
    )

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(
        f"Optimal value: {f_opt}\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret{regret}"
    )
    return agent


def main():
    logging.warning(
        "This main method is for illustrative purposes only and will NEVER be called by the checker!\n"
        "The checker always calls run_solution directly.\n"
        "Please implement your solution exclusively in the methods and classes mentioned in the task description."
    )

    seed = 1234
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = BO_algo()

    agent = train_on_toy(agent, 20)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(agent)


if __name__ == "__main__":
    main()
