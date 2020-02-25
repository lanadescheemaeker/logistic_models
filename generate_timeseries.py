import numpy as np
import pandas as pd
from brownian import brownian
from noise_parameters import NOISE, MODEL
import time

def make_params(steadystate,
                interaction=0, selfint=1,
                immigration=0, init_dev=0.1,
                noise=1e-1, connectivity=1):
    """
    Return set of parameters: interaction matrix, growth rate, immigration rate, noise and initial condition

    :param steadystate: steady state, list of floats
    :param interaction: interaction strength, float
    :param selfint: self interaction, float or list of floats
    :param immigration: immigration rate, float or list of floats
    :param init_dev: deviation from steady state for initial condition, float
    :param noise: amount of noise, float
    :param connectivity: connectivity, float between 0 and 1
    :return: dictionary of parameters
    """
    params = {}

    n = len(steadystate)

    if interaction == 0:
        omega = np.zeros([n, n]);
    else:
        omega = np.random.normal(0, interaction, [n,n])
        omega *= np.random.choice([0, 1], omega.shape, p=[1-connectivity, connectivity])
    np.fill_diagonal(omega, -selfint)

    params['interaction_matrix'] = omega

    params['immigration_rate'] = np.full(steadystate.shape, immigration)

    # different growth rates determined by the steady state
    params['growth_rate'] = - (omega).dot(steadystate)

    params['initial_condition'] = np.copy(steadystate) * np.random.normal(1, init_dev, steadystate.shape)

    params['noise_linear'] = noise
    params['noise_sqrt'] = noise
    params['noise_constant'] = noise

    return params

def Jacobian(intmat, ss, K):
    J = intmat * ss.reshape([np.prod(ss.shape), 1])

    return J

# TODO check shape of steadystate
def is_stable(steadystate, interaction_matrix):
    """
    Checks whether steady state is stable solution of generalized Lotka Volterra system with given interaction matrix

    :param steadystate: np.array with steady state
    :param interaction_matrix: np.array with interaction matrix
    :return: bool: stability of steady state
    """

    # Jacobian
    Jac = interaction_matrix * steadystate

    if np.any(np.real(np.linalg.eigvals(Jac)) > 0):
        return False
    else:
        return True

def test_validity_Jacobian():
    """
    Test to check whether the Jacobian used in the is_stable function is correctly defined
    Print statements
    :return: bool if definition is correct
    """

    def numeric_Jacobian(intmat, ss, K):
        epsilon = 1e-5

        def f(x):
            return x * (K + np.dot(intmat, x))

        J = np.zeros(intmat.shape)

        for i in range(len(J)):
            dx = np.zeros(ss.shape);
            dx[i] = epsilon
            J[:, i] = ((f(ss + dx) - f(ss - dx)) / (2 * epsilon)).flatten()

        return J

    n = 6

    ss = np.random.uniform(0, 5, [n, 1])  # np.ones([N,1])
    intmat = np.random.normal(0, 3, [n, n])
    K = - np.dot(intmat, ss)

    Jac = Jacobian(intmat, ss, K)
    numJac = numeric_Jacobian(intmat, ss, K)

    print("Jacobian", Jac)
    print("numeric Jacobian", numJac)
    print("maximal relative difference", np.max(abs((numJac - Jac)/Jac)))
    return np.max(abs((numJac - Jac)/Jac)) < 1e-6

class Timeseries():
    def __init__(self, params, model=MODEL.GLV, noise_implementation=NOISE.LANGEVIN_LINEAR, dt=0.01, T=100, tskip=0,
                 f=0, seed=None):
        self.params = params
        self.model = model
        self.noise_implementation = noise_implementation
        self.dt = dt
        self.T = T
        self.tskip = tskip
        self.f = f
        self.seed = seed

        self.set_seed()

        self.check_input_parameters()

        self.init_Nspecies_Nmetabolites()

        self.deterministic_step = self.deterministic_step_function()
        self.stochastic_step = self.stochastic_step_function()
        self.add_step = self.add_step_function()

        if self.model == MODEL.GLV:
            self.x = np.copy(self.params['initial_condition'])
        elif self.model == MODEL.QSMI:
            self.x = np.copy(self.params['initial_condition'])[:len(self.params['d'])]  # initial state species
            self.y = np.copy(self.params['initial_condition'])[len(self.params['d']):]  # initial state metabolites

        self.x_ts = np.copy(self.x)

        if f != 0:
            self.write_header()

        self.integrate()

    def set_seed(self):
        if self.seed == None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.seed)

    def check_input_parameters(self):
        # Function to check if all necessary parameters where provided, raises error if parameters are missing

        if self.model == MODEL.GLV:
            parlist = ['interaction_matrix', 'immigration_rate', 'growth_rate', 'initial_condition']

            if 'LINEAR' in self.noise_implementation.name:
                parlist += ['noise_linear']
            elif 'SQRT' in self.noise_implementation.name:
                parlist += ['noise_sqrt']
            elif 'CONSTANT' in self.noise_implementation.name:
                parlist += ['noise_constant']
            elif 'INTERACTION' in self.noise_implementation.name:
                parlist += ['noise_interaction']

        elif self.model == MODEL.QSMI:
            parlist = ['psi', 'd', 'g', 'dm', 'kappa', 'phi', 'initcond', 'noise']

        for par in parlist:
            if not par in self.params:
                raise KeyError('Parameter %s needs to be specified for the %s model and %s noise implementation.' % (
                    par, self.model.name, self.noise_implementation.name))

        # check whether matrix shapes are correct
        if self.model == MODEL.GLV:
            if not np.all(len(row) == len(self.params['interaction_matrix']) for row in
                          self.params['interaction_matrix']):
                raise ValueError('Interaction matrix is not square.')

            for parname in ['immigration_rate', 'growth_rate', 'initial_condition']:
                if np.any(self.params[parname].shape != (self.params['interaction_matrix'].shape[0], 1)):
                    raise ValueError('%s has the incorrect shape: %s instead of (%d,1)' % (
                        parname, str(self.params[parname].shape), self.params['interaction_matrix'].shape[0]))

    def write_header(self):
        # Write down header in file f
        with open(self.f, "a") as file:
            file.write("time")
            for k in range(1, self.Nspecies + 1):
                file.write(",species_%d" % k)
            for k in range(1, self.Nmetabolites + 1):
                file.write(",metabolite_%d" % k)

            file.write("\n")

            file.write("%.3E" % 0)
            for k in self.params['initial_condition']:
                file.write(",%.3E" % k)
            file.write("\n")

    def init_Nspecies_Nmetabolites(self):
        if self.model == MODEL.GLV:
            self.Nspecies = len(self.params['interaction_matrix'])  # number of species
            self.Nmetabolites = 0  # number of metabolites, 0 in the GLV models
        elif self.model == MODEL.QSMI:
            self.Nspecies = len(self.params['d'])  # number of species
            self.Nmetabolites = len(self.params['dm'])  # number of metabolites

    def integrate(self):
        # If noise is Ito, first generate brownian motion.
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt = np.zeros_like(self.params['initial_condition'])
            self.bm = brownian(np.zeros(len(self.params['initial_condition'])), int(self.T / self.dt), self.dt, 1,
                               out=None)

        x_ts = np.zeros([int(self.T / (self.dt * (self.tskip + 1))), self.Nspecies])

        # set initial condition
        x_ts[0] = self.x.flatten()

        # Integrate ODEs according to model and noise
        for i in range(1, int(self.T / (self.dt * (self.tskip + 1)))):
            for j in range(self.tskip + 1):
                self.add_step(self, i * (self.tskip + 1) + j)

            # Save abundances
            if self.f != 0:
                self.write_abundances_to_file(i * (self.tskip + 1) + j)

            x_ts[i] = self.x.flatten()

            if np.all(np.isnan(self.x)):
                break

        # dataframe to save timeseries
        self.x_ts = pd.DataFrame(x_ts, columns=['species_%d' % i for i in range(1, self.Nspecies + 1)])
        self.x_ts['time'] = (self.dt * (self.tskip + 1) * np.arange(0, int(self.T / (self.dt * (self.tskip + 1)))))

        return

    def add_step_function(self):
        if self.model == MODEL.GLV:
            if ('LANGEVIN' in self.noise_implementation.name or 'MILSTEIN' in self.noise_implementation.name):
                def func(self, i):
                    # print(type(deterministic_step))

                    dx_det = self.deterministic_step(self)
                    dx_stoch = self.stochastic_step(self)

                    self.x += dx_det + dx_stoch

                    # abundance cannot be negative
                    self.x = self.x.clip(min=0)

            elif 'RICKER' in self.noise_implementation.name:
                def func(self, i):
                    self.ricker_step()

            elif 'ARATO' in self.noise_implementation.name:
                def func(self, i):
                    self.arato_step(i)

        elif self.model == MODEL.QSMI:
            def func(self, i):
                dx_det, dy_det = self.deterministic_step()

                # TODO implement the stochastic version of QMSI

                self.x += dx_det
                self.y += dy_det

                self.x = self.x.clip(min=0)
                self.y = self.y.clip(min=0)

        return func

    def deterministic_step_function(self):
        if self.model == MODEL.GLV:
            def func(self):
                return (self.params['interaction_matrix'].dot(self.x) * self.x + self.params['immigration_rate'] +
                        self.params['growth_rate'] * self.x) * self.dt

        elif self.model == MODEL.QSMI:
            if self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                def func(self):
                    dx = self.x * (self.params['psi'].dot(self.y) - self.params['d']) * self.dt
                    dy = (self.params['g'] - self.params['dm'] * self.y - self.y * self.params['kappa'].dot(self.x) + (
                    (self.params['phi'].dot(self.x)).reshape([self.Nmetabolites, self.Nmetabolites])).dot(
                        self.y)) * self.dt
                    return dx, dy

        return func

    def stochastic_step_function(self):
        if self.model == MODEL.GLV:
            if self.noise_implementation == NOISE.LANGEVIN_LINEAR:
                def func(self):
                    return self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1,
                                                                                                      self.x.shape)
            elif self.noise_implementation == NOISE.GROWTH_AND_INTERACTION_LINEAR:
                def func(self):
                    return (
                    self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape) + (
                    self.params['noise_interaction'] * np.random.normal(0, 1,
                                                                        self.params['interaction_matrix'].shape)).dot(
                        self.x) * self.x * np.sqrt(self.dt))
            elif self.noise_implementation == NOISE.LANGEVIN_SQRT:
                def func(self):
                    return self.params['noise_sqrt'] * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.normal(0, 1,
                                                                                                             self.x.shape)
            elif self.noise_implementation == NOISE.LANGEVIN_LINEAR_SQRT:
                def func(self):
                    return self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1,
                                                                                                      self.x.shape) + \
                           self.params['noise_sqrt'] * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.normal(0, 1,
                                                                                                             self.x.shape)
            elif self.noise_implementation == NOISE.SQRT_MILSTEIN:
                def func(self):
                    dW = np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)
                    return np.sqrt(self.params['noise_sqrt'] * self.x) * dW + self.params['noise_sqrt'] ** 2 / 4 * (
                        dW ** 2 - self.dt ** 2)

            elif self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                def func(self):
                    return self.params['noise_constant'] * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)

            return func

    def ricker_step(self):
        if self.noise_implementation == NOISE.RICKER_LINEAR:
            if self.params['noise_linear'] == 0:
                b = np.ones(self.x.shape)
            else:
                b = np.exp(self.params['noise_linear'] * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape))
            self.x = b * self.x * np.exp(self.params['interaction_matrix'].dot(
                self.x + np.linalg.inv(self.params['interaction_matrix']).dot(self.params['growth_rate'])) * self.dt)
        else:
            raise ValueError('No implementation for "%s"' % self.noise_implementation.name)

    def arato_step(self, i):
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt += self.x * self.dt

            t = i * self.dt

            Y = self.params['growth_rate'] * t - self.params['noise_linear'] ** 2 / 2 * t + self.params[
                'interaction_matrix'].dot(self.xt) + self.params['noise_linear'] * self.bm[:, i].reshape(
                self.x.shape)  # noise * np.random.normal(0, 1, initcond.shape)
            self.x = self.params['initial_condition'] * np.exp(Y)

    def write_abundances_to_file(self, i):
        with open(self.f, "a") as file:
            file.write("%.5E" % (i * self.dt))
            for k in self.x:
                file.write(",%.5E" % k)
            if self.model == MODEL.QSMI:
                for k in self.y:
                    file.write(",%.5E" % k)
            file.write("\n")

    @property
    def timeseries(self):
        return self.x_ts

    @property
    def endpoint(self):
        return self.x

def test_timeseries():
    print('test Timeseries')

    N = 50

    params = {}

    steadystate = np.logspace(-3, 2, N).reshape([N, 1])

    # no interaction
    omega = np.zeros([N, N]);
    np.fill_diagonal(omega, -1)

    params['interaction_matrix'] = omega

    # no immigration
    params['immigration_rate'] = np.zeros([N, 1])

    # different growth rates determined by the steady state
    params['growth_rate'] = - (omega).dot(steadystate)

    params['initial_condition'] = np.copy(steadystate) * np.random.normal(1, 0.1, steadystate.shape)

    params['noise_linear'] = 1e-1

    ts = Timeseries(params, noise_implementation=NOISE.LANGEVIN_LINEAR, dt=0.01, tskip=4, T=100.0,
                    seed=int(time.time()))

    print("timeseries")
    print(ts.timeseries.head())

    print("endpoint")
    print(ts.endpoint)

    return ts

def main():
    #test_validity_Jacobian()
    test_timeseries()

if __name__ == "__main__":
    main()