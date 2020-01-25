import numpy as np
import pandas as pd
from brownian import *
from noise_parameters import *
import time

def make_params(steadystate,
                interaction=0, selfint=1,
                immigration=0, init_dev=0.1,
                noise=1e-1, connectivity=1):
    params = {}

    N = len(steadystate)

    if interaction == 0:
        omega = np.zeros([N, N]);
    else:
        omega = np.random.normal(0, interaction, [N,N])
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

def is_stable(steadystate, interaction_matrix):
    # Jacobian

    Jac = interaction_matrix * steadystate

    if np.any(np.real(np.linalg.eigvals(Jac)) > 0):
        return False
    else:
        return True

def test_validity_Jacobian():
    def Jacobian(intmat, ss, K):
        J = intmat * ss  # + np.diag(K + np.diag(intmat)*ss.flatten())

        return J

    def numJac(intmat, ss, K):
        epsilon = 1e-5

        def f(x):
            return x * (K + np.dot(intmat, x))

        J = np.zeros(intmat.shape)

        for i in range(len(J)):
            dx = np.zeros(ss.shape);
            dx[i] = epsilon
            J[:, i] = ((f(ss + dx) - f(ss - dx)) / (2 * epsilon)).flatten()

        return J

    N = 6

    ss = np.random.uniform(0, 5, [N, 1])  # np.ones([N,1])
    intmat = np.random.normal(0, 3, [N, N])
    K = - np.dot(intmat, ss)

    print(Jacobian(intmat, ss, K))
    print(numJac(intmat, ss, K))
    print(intmat * ss)
    print(numJac(intmat, ss, K) - intmat)

class Timeseries():
    def __init__(self, params, model = MODEL.GLV, noise_implementation=NOISE.LANGEVIN_LINEAR,
                         dt=0.01, T=100, tskip = 0,
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

        if self.model == MODEL.GLV:
            self.x = np.copy(self.params['initial_condition'])
        elif self.model == MODEL.QSMI:
            self.x = np.copy(self.params['initial_condition'])[:len(self.params['d'])]  # initial state species
            self.y = np.copy(self.params['initial_condition'])[len(self.params['d']):]  # initial state metabolites

        self.x_ts = np.copy(self.x)

        if f!= 0:
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
            if not np.all(len(row) == len(self.params['interaction_matrix']) for row in self.params['interaction_matrix']):
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
            self.Nspecies = len(self.params['interaction_matrix']) # number of species
            self.Nmetabolites = 0 # number of metabolites, 0 in the GLV models
        elif self.model == MODEL.QSMI:
            self.Nspecies = len(self.params['d']) # number of species
            self.Nmetabolites = len(self.params['dm']) # number of metabolites

    def integrate(self):
        # If noise is Ito, first generate brownian motion.
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt = np.zeros_like(self.params['initial_condition'])
            self.bm = brownian(np.zeros(len(self.params['initial_condition'])),
                               int(self.T /self.dt), self.dt, 1, out=None)

        # Integrate ODEs according to model and noise
        for i in range(1, int(self.T / self.dt)):
            self.add_step(i)

            # Save abundances
            if self.f != 0 and i % (self.tskip + 1) == 0:
                self.write_abundances_to_file(i)

            if i % (self.tskip + 1) == 0:
                self.x_ts = np.hstack((self.x_ts, self.x))

        self.x_ts = np.vstack((self.dt * (self.tskip + 1) * np.arange(len(self.x_ts[0]))[np.newaxis, :], self.x_ts))
        self.x_ts = pd.DataFrame(self.x_ts.T, columns=['time'] + ['species_%d' % i for i in range(1, self.Nspecies + 1)])

    def add_step(self, i):
        if self.model == MODEL.GLV:
            if ('LANGEVIN' in self.noise_implementation.name
                or 'MILSTEIN' in self.noise_implementation.name):

                dx_det = self.deterministic_step()
                dx_stoch = self.stochastic_step()

                self.x += dx_det + dx_stoch

                # abundance cannot be negative
                self.x = self.x.clip(min=0)

            elif 'RICKER' in self.noise_implementation.name:
                self.ricker_step()

            elif 'ARATO' in self.noise_implementation.name:
                self.arato_step(i)

        elif self.model == MODEL.QSMI:

            dx_det, dy_det = self.deterministic_step()

            # TODO implement the stochastic version of QMSI

            self.x += dx_det
            self.y += dy_det

            self.x = self.x.clip(min=0)
            self.y = self.y.clip(min=0)

    def deterministic_step(self):
        if self.model == MODEL.GLV:
            dx = (self.params['interaction_matrix'].dot(self.x) * self.x + self.params['immigration_rate'] +
                  self.params['growth_rate'] * self.x) * self.dt

            return dx
        elif self.model == MODEL.QSMI:
            if self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                dx = self.x * (self.params['psi'].dot(self.y) - self.params['d']) * self.dt
                dy = (self.params['g'] - self.params['dm'] * self.y - self.y * self.params['kappa'].dot(self.x) + \
                     ((self.params['phi'].dot(self.x)).reshape([self.Nmetabolites, self.Nmetabolites])).dot(self.y)) * self.dt
            return dx, dy

    def stochastic_step(self):
        if self.model == MODEL.GLV:
            if self.noise_implementation == NOISE.LANGEVIN_LINEAR:
                dx = self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)
            elif self.noise_implementation == NOISE.GROWTH_AND_INTERACTION_LINEAR:
                dx = (self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape) + \
                     (self.params['noise_interaction'] *
                      np.random.normal(0, 1, self.params['interaction_matrix'].shape)).dot(self.x) \
                     * self.x * np.sqrt(self.dt))
            elif self.noise_implementation == NOISE.LANGEVIN_SQRT:
                dx = self.params['noise_sqrt'] * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)
            elif self.noise_implementation == NOISE.LANGEVIN_LINEAR_SQRT:
                dx = self.params['noise_linear'] * self.x * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape) + \
                     self.params['noise_sqrt'] * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.normal(0, 1,
                                                                                                       self.x.shape)
            elif self.noise_implementation == NOISE.SQRT_MILSTEIN:
                dW = np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)
                dx = np.sqrt(self.params['noise_sqrt'] * self.x) * dW + self.params['noise_sqrt'] ** 2 / 4 * (
                        dW ** 2 - self.dt ** 2)
            elif self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                dx = self.params['noise_constant'] * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape)

            return dx

    def ricker_step(self):
        if self.noise_implementation == NOISE.RICKER_LINEAR:
            if self.params['noise_linear'] == 0:
                b = np.ones(self.x.shape)
            else:
                b = np.exp(self.params['noise_linear'] * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape))
            self.x = b * self.x * np.exp(
                self.params['interaction_matrix'].dot(self.x
                        + np.linalg.inv(self.params['interaction_matrix']).dot(self.params['growth_rate'])) * self.dt)
        else:
            raise ValueError('No implementation for "%s"' % self.noise_implementation.name)

    def arato_step(self, i):
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt += self.x * self.dt

            t = i * self.dt

            Y = self.params['growth_rate'] * t - self.params['noise_linear'] ** 2 / 2 * t \
                + self.params['interaction_matrix'].dot(self.xt) + self.params['noise_linear'] * self.bm[:, i].reshape(
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


def main():
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

    ts = Timeseries(params, noise_implementation=NOISE.LANGEVIN_LINEAR,
                    dt=0.01, tskip=4, T=100.0, seed=int(time.time()))

    print("timeseries")
    print(ts.timeseries.head())

    print("endpoint")
    print(ts.endpoint)

    return ts

if __name__ == "__main__":
    main()

# old code

""" 
def check_input_parameters(params, model, noise_implementation):
    # Function to check if all necessary parameters where provided, raises error if parameters are missing

    if model == MODEL.GLV:
        parlist = ['interaction_matrix', 'immigration_rate', 'growthrate', 'initial_condition']

        if 'LINEAR' in noise_implementation.name:
            parlist += ['noise_linear']
        elif 'SQRT' in noise_implementation.name:
            parlist += ['noise_sqrt']
        elif 'INTERACTION' in noise_implementation.name:
            parlist += ['noise_interaction']

    elif model == MODEL.QSMI:
        parlist = ['psi', 'd', 'g', 'dm', 'kappa', 'phi', 'initcond', 'noise']

    for par in parlist:
        if not par in params:
            raise KeyError('Parameter %s needs to be specified for the %s model and %s noise implementation.' % (
                par, model.name, noise_implementation.name))

    # check whether matrix shapes are correct

    if model == MODEL.GLV:
        if not np.all(len(row) == len(params['interaction_matrix']) for row in params['interaction_matrix']):
            raise ValueError('Interaction matrix is not square.')

        for parname in ['immigration_rate', 'growthrate', 'initial_condition']:
            if np.any(params[parname].shape != (params['interaction_matrix'].shape[0], 1)):
                raise ValueError('%s has the incorrect shape: %s instead of (%d,1)'
                                 % (parname, str(params[parname].shape), params['interaction_matrix'].shape[0]))


def run_timeseries_noise(params, model = MODEL.GLV, noise_implementation=NOISE.LANGEVIN_LINEAR,
                         dt=0.01, T=100, tskip = 0,
                         f=0, ts = True, seed=None):

    # Set seed for random number generator
    if seed == None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    # Verify if all parameters are given, otherwise raise error.
    check_input_parameters(params, model, noise_implementation)

    # Set parameters.
    if model == MODEL.GLV:
        omega, mu, g, initcond = params['interaction_matrix'], params['immigration_rate'], \
                                     params['growthrate'], params['initial_condition']

        for noise in ['noise_linear', 'noise_sqrt', 'noise_interaction']:
            if noise in params:
                locals()[noise] = params[nois]

        Nspecies = len(omega) # number of species
        Nmetabolites = 0 # number of metabolites, 0 in the GLV models
        x = np.copy(initcond) # set initial state
    elif model == MODEL.QSMI:
        psi, d, g, dm, kappa, phi, initcond, noise =  params['psi'], params['d'], params['g'], params['dm'], \
                                                   params['kappa'], params['phi'], params['initcond'], params['noise']

        Nspecies = len(d) # number of species
        Nmetabolites = len(dm) # number of metabolites
        x = np.copy(initcond)[:len(d)] # initial state species
        y = np.copy(initcond)[len(d):] # initial state metabolites

    # Write down header in file
    if f != 0:
        with open(f, "a") as file:
            file.write("time")
            for k in range(1, Nspecies + 1):
                file.write(",species_%d" % k)
            for k in range(1, Nmetabolites + 1):
                file.write(",metabolite_%d" % k)

            file.write("\n")

            file.write("%.3E" % 0)
            for k in initcond:
                file.write(",%.3E" % k)
            file.write("\n")

    # To save all points in timeseries, make new variable x_ts
    if ts == True:
        x_ts = np.copy(x)

    # If noise is Ito, first generate brownian motion.
    if noise_implementation == NOISE.ARATO_LINEAR:
        xt = np.zeros_like(initcond)
        bm = brownian(np.zeros(len(initcond)), int(T /dt), dt, 1, out=None)

    # Integrate ODEs according to model and noise
    for i in range(1, int(T / dt)):
        if model == MODEL.GLV:
            if noise_implementation == NOISE.LANGEVIN_LINEAR:
                dx = (omega.dot(x) * x + mu + g * x) * dt
                x += dx + noise * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.GROWTH_AND_INTERACTION_LINEAR:
                dx = (omega.dot(x) * x + mu + g * x) * dt
                x += dx + noise_linear * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape) \
                     + (noise_interaction * np.random.normal(0, 1, omega.shape)).dot(x) * x * np.sqrt(dt)
            elif noise_implementation == NOISE.LANGEVIN_SQRT:
                x += (omega.dot(x) * x + mu + g * x) * dt + noise * np.sqrt(x) * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.LANGEVIN_LINEAR_SQRT:
                x += (omega.dot(x) * x + mu + g * x) * dt + noise_linear * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape) \
                     + noise_sqrt * np.sqrt(x) * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.SQRT_MILSTEIN:
                dW = np.sqrt(dt)*np.random.normal(0,1, x.shape)
                x += (omega.dot(x) * x + mu + g * x) * dt + np.sqrt(noise * x) * dW + noise**2 / 4 * (dW**2 - dt**2)
            elif noise_implementation == NOISE.LANGEVIN_CONSTANT:
                x += (omega.dot(x) * x + mu + g * x) * dt + noise * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.RICKER_LINEAR:
                if noise == 0:
                    b = np.ones(x.shape)
                else:
                    b = np.exp(noise * np.sqrt(dt) * np.random.normal(0, 1, x.shape))
                x = b * x * np.exp(omega.dot(x + np.linalg.inv(omega).dot(g)) * dt)
            elif noise_implementation == NOISE.ARATO_LINEAR:
                xt += x * dt

                t = i * dt

                Y = g * t - noise ** 2 / 2 * t + omega.dot(xt) + noise * bm[:, i].reshape(
                    x.shape)  # noise * np.random.normal(0, 1, initcond.shape)
                x = initcond * np.exp(Y)

            x = x.clip(min=0)

        if model == MODEL.QSMI:
            if noise_implementation == NOISE.LANGEVIN_CONSTANT:
                dx = x*(psi.dot(y) - d)
                dy = g - dm*y - y*kappa.dot(x) + ((phi.dot(x)).reshape([Nmetabolites,Nmetabolites])).dot(y)

                x += dx*dt
                y += dy*dt #+ noise * np.sqrt(dt) * np.random.normal(0, 1, y.shape)

                x = x.clip(min=0)
                y = y.clip(min=0)

        # Save abundances
        if f != 0 and i % (tskip + 1) == 0:
            with open(f, "a") as file:
                file.write("%.5E" % (i * dt))
                for k in x:
                    file.write(",%.5E" % k)
                if model == MODEL.QSMI:
                    for k in y:
                        file.write(",%.5E" % k)
                file.write("\n")
        if ts == True and i % (tskip + 1) == 0:
            x_ts = np.hstack((x_ts, x))

    # return timeseries if ts = True, else return only endpoint
    if ts == True:
        x_ts = np.vstack((dt * (tskip+1)*np.arange(len(x_ts[0]))[np.newaxis, :], x_ts))
        x_ts = pd.DataFrame(x_ts.T, columns=['time'] + ['species_%d' % i for i in range(1,Nspecies+1)])
        return x_ts
    else:
        return x

def generate_timeseries_noise(par=[0, 0, 0], fts=None, fomega=None, fg=None, fmu=None, SIS=False, noise=0.1, noise_implementation=NOISE.LANGEVIN_SQRT):
    connectance, intdiv, r = par

    np.random.seed(r)

    S = 100  # 40
    distromega = Distribution.NORMAL  # UNIFORM #NORMAL
    distrgrowth = Distribution.UNIFORM
    growthrate = 1
    minmigration = 0
    maxmigration = 0  # 0.1

    SISvector = np.ones(S)  # 0.001

    if SIS:
        SISvector[0] = 200

    initcond = np.zeros(S)

    # get variables if given in files

    omega_given, mu_given, g_given = False, False, False

    if isinstance(fomega, np.ndarray):
        omega = fomega
        omega_given = True
    elif fomega != None:
        if isinstance(fomega, str) and os.path.exists(fomega):
            if fomega.endswith('.csv'):
                omega = pd.read_csv(fomega, index_col=0).values
                omega_given = True
            elif fomega.endswith('.txt'):
                omega = np.loadtxt(fomega)
                omega_given = True

    if fmu != None:
        if os.path.exists(fmu):
            mu = pd.read_csv(fmu, index_col=0).values
            mu_given = True

    if isinstance(fg, np.ndarray):
        g = fg.reshape([S, 1])
        g_given = True
    elif fg != None:
        if isinstance(fg, str) and os.path.exists(fg):
            if fomega.endswith('.csv'):
                g = pd.read_csv(fg, index_col=0).values.reshape([S,1])
                g_given = True
            elif fomega.endswith('.txt'):
                g = np.loadtxt(fg).reshape([S,1])
                g_given = True

    # Look for non-zero steady state

    while np.sum(initcond != 0) < 2 or np.any(np.isnan(initcond)) or (SIS and initcond[0]==0):
        # get new variables if not yet given

        if not omega_given:
            omega = interactionmatrix(S, intdiv, distromega, connectance, SISvector)

        if not mu_given:
            mu = np.random.uniform(minmigration, maxmigration, [S, 1])

        if not g_given:
            if distrgrowth == Distribution.CONSTANT:
                g = np.full([S, 1], growthrate)
            elif distrgrowth == Distribution.UNIFORM:
                g = np.random.uniform(0, growthrate, [S, 1])

        initcond = np.random.uniform(0, 1, [S,1])

        dspecies = omega.dot(initcond) * initcond + mu + g * initcond

        while not (np.all(initcond == 0) or np.any(np.isnan(initcond))) and np.nanmax(
                abs((dspecies / initcond)[initcond != 0])) > 1e-1: #1e-3:
            if np.all(initcond == 0):
                print("Initial condition is zero for all species.")
            elif np.any(np.isnan(initcond)):
                print("The initial condition of one of the species is nan.")
            else:
                print("Not yet in steady state, maximum absolute derivative is", np.nanmax(
                abs((dspecies / initcond)[initcond != 0])))

            initcond = run_timeseries(omega, mu, g, initcond) #, f='testje.csv')

            #plot_timeseries('testje.csv')
            #os.remove('testje.csv')
            #plt.show()

            initcond[initcond < 1e-30] = 0

            dspecies = omega.dot(initcond) * initcond + mu + g * initcond

    # Perform stochastic simulation.

    if omega_given == False and fomega != None:
        pd.DataFrame(omega).to_csv(fomega)
    if mu_given == False and fmu != None:
        pd.DataFrame(mu).to_csv(fmu)
    if g_given == False and fg != None:
        pd.DataFrame(g).to_csv(fg)

    run_timeseries_Langevin(omega, mu, g, initcond, noise, fts, noise_implementation)

def generate_timeseries_noise_loop_parameters():
    for i, ii in zip([0.1, 0.25, 0.4, 0.6, 0.8], [0, 2, 5, 7, 1]):
        for j, jj in zip([0.02, 0.05, 0.1, 0.15, 0.2], [0, 2, 5, 7, 1]):
            for l in ['a', 'b', 'c', 'd', 'e']:
                #print('glv/noise/data_SIS_wide_%d%d%s.csv' % (ii, jj, l))
                if not os.path.isfile('glv/test/data_SIS_wide_%d%d%s.csv' % (ii, jj, l)):
                    fts = 'glv/test/data_SIS_%d%d%s.csv' % (ii, jj, l)
                    fomega = 'glv/test/omega_SIS_%d%d%s.csv' % (ii, jj, l)
                    fmu = 'glv/test/mu_SIS_%d%d%s.csv' % (ii, jj, l)
                    fg = 'glv/test/g_SIS_%d%d%s.csv' % (ii, jj, l)
                    generate_timeseries_noise([i, j, int(time.time())], fts, fomega, fg, fmu, SIS=True)

"""