#!/usr/bin/env python3

import theano
import warnings
import numpy as np
import pymc3 as pm
import arviz as az
import xarray as xr
import theano.tensor as tt
import matplotlib.pyplot as plt

warnings.filterwarnings(action="ignore")
theano.config.compute_test_value = "warn"
# theano.config.mode = "JAX"


class PDDModel(object):
    """

    # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
    # GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

    A positive degree day model for glacier surface mass balance

    Return a callable Positive Degree Day (PDD) model instance.

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(
        self,
        pdd_factor_snow=0.003,
        pdd_factor_ice=0.008,
        refreeze_snow=0.0,
        refreeze_ice=0.0,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=52,
        *args,
        **kwargs,
    ):
        super().__init__()

        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    def __call__(self, temp, prec, stdv=0.0):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.

        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.

        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        temp = self._interpolate(temp)
        prec = self._interpolate(prec)
        stdv = self._interpolate(stdv)

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth, melt and refreeze rates
        snow_depth = np.zeros_like(temp)
        snow_melt_rate = np.zeros_like(temp)
        ice_melt_rate = np.zeros_like(temp)
        snow_refreeze_rate = np.zeros_like(temp)
        ice_refreeze_rate = np.zeros_like(temp)

        # compute snow depth and melt rates
        for i in range(len(temp)):
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accu_rate[i]
            snow_melt_rate[i], ice_melt_rate[i] = self.melt_rates(
                snow_depth[i], inst_pdd[i]
            )
            snow_depth[i] -= snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
            "inst_pdd": inst_pdd,
            "accu_rate": accu_rate,
            "snow_melt_rate": snow_melt_rate,
            "ice_melt_rate": ice_melt_rate,
            "melt_rate": melt_rate,
            "snow_refreeze_rate": snow_refreeze_rate,
            "ice_refreeze_rate": ice_refreeze_rate,
            "refreeze_rate": refreeze_rate,
            "runoff_rate": runoff_rate,
            "inst_smb": inst_smb,
            "snow_depth": snow_depth,
            "pdd": self._integrate(inst_pdd),
            "accu": self._integrate(accu_rate),
            "snow_melt": self._integrate(snow_melt_rate),
            "ice_melt": self._integrate(ice_melt_rate),
            "melt": self._integrate(melt_rate),
            "runoff": self._integrate(runoff_rate),
            "refreeze": self._integrate(refreeze_rate),
            "smb": self._integrate(inst_smb),
        }

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * np.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return np.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""
        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (np.arange(len(array) + 2) - 0.5) / len(array)
        oldy = np.vstack(([array[-1]], array, [array[0]]))
        newx = (np.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
        return newy

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.

        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """
        import scipy.special as sp

        # compute positive part of temperature everywhere
        positivepart = np.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = temp / (np.sqrt(2) * stdv)
        calovgreve = stdv / np.sqrt(2 * np.pi) * np.exp(
            -(normtemp**2)
        ) + temp / 2 * sp.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = np.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accu_rate(self, temp, prec):
        """Compute accumulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain - temp) / (self.temp_rain - self.temp_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac * prec

    def melt_rates(self, snow, pdd):
        """Compute melt rates from snow precipitation and pdd sum.

        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.

        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days.
        """

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow
        ddf_ice = self.pdd_factor_ice

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class TTPDDModel(object):
    """

    # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
    # GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

    A positive degree day model for glacier surface mass balance

    Return a callable Positive Degree Day (PDD) model instance.

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(
        self,
        pdd_factor_snow=0.003,
        pdd_factor_ice=0.008,
        refreeze_snow=0.0,
        refreeze_ice=0.0,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=52,
        *args,
        **kwargs,
    ):
        super().__init__()

        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    def __call__(self, temp, prec, stdv=0.0):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.

        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.

        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        temp = self._interpolate(temp)
        prec = self._interpolate(prec)
        stdv = self._interpolate(stdv)

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth, melt and refreeze rates
        snow_depth = theano.shared(np.zeros_like(temp))
        snow_melt_rate = theano.shared(np.zeros_like(temp))
        ice_melt_rate = theano.shared(np.zeros_like(temp))
        snow_refreeze_rate = theano.shared(np.zeros_like(temp))
        ice_refreeze_rate = theano.shared(np.zeros_like(temp))

        # compute snow depth and melt rates

        for i in range(len(temp)):
            if i > 0:
                snow_depth = tt.set_subtensor(snow_depth[i], snow_depth[i - 1])
            snow_depth = tt.inc_subtensor(snow_depth[i], accu_rate[i])
            smr, imr = self.melt_rates(snow_depth[i], inst_pdd[i])
            snow_melt_rate = tt.set_subtensor(snow_melt_rate[i], smr[i])
            ice_melt_rate = tt.set_subtensor(ice_melt_rate[i], imr[i])
            snow_depth = tt.inc_subtensor(snow_depth[i], -snow_melt_rate[i])

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
            "inst_pdd": inst_pdd,
            "accu_rate": accu_rate,
            "snow_melt_rate": snow_melt_rate,
            "ice_melt_rate": ice_melt_rate,
            "melt_rate": melt_rate,
            "snow_refreeze_rate": snow_refreeze_rate,
            "ice_refreeze_rate": ice_refreeze_rate,
            "refreeze_rate": refreeze_rate,
            "runoff_rate": runoff_rate,
            "inst_smb": inst_smb,
            "snow_depth": snow_depth,
            "pdd": self._integrate(inst_pdd),
            "accu": self._integrate(accu_rate),
            "snow_melt": self._integrate(snow_melt_rate),
            "ice_melt": self._integrate(ice_melt_rate),
            "melt": self._integrate(melt_rate),
            "runoff": self._integrate(runoff_rate),
            "refreeze": self._integrate(refreeze_rate),
            "smb": self._integrate(inst_smb),
        }

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * np.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return tt.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""
        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (np.arange(len(array) + 2) - 0.5) / len(array)
        oldy = np.vstack(([array[-1]], array, [array[0]]))
        newx = (np.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
        return newy

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.

        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """

        # compute positive part of temperature everywhere
        positivepart = tt.gt(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = temp / (np.sqrt(2) * stdv)
        calovgreve = stdv / np.sqrt(2 * np.pi) * tt.exp(
            -(normtemp**2)
        ) + temp / 2 * tt.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = tt.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accu_rate(self, temp, prec):
        """Compute accumulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain - temp) / (self.temp_rain - self.temp_snow)
        snowfrac = tt.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac * prec

    def melt_rates(self, snow, pdd):
        """Compute melt rates from snow precipitation and pdd sum.

        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.

        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days.
        """

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = tt.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class PDD_MCMC:
    def __init__(self):
        # These a arguments which all models will need. Any of the model
        # pamaeters which are dependent on the model formulation are passed
        # as kwargs.

        pass

    def forward(self, temp, precip, std_dev, f_snow, f_ice, f_refreeze):
        """Theano implementation of the forward model which supports shared
           variables as input.

           This is an intermidiate function which returns a list of the three
           components of the PDD model (i.e. Accumulation, Refreezing and melt).

        Inputs:
            temp          (ndarray) Nx1 array of temperature  [K]
            precip        (ndarray) Nx1 array of precipitatoin  [m / yr]
            f_snow        (float)   degree-day factor for snow      [kg m^-2 yr^-1 K^-1]
            f_ice         (float)   degree-day factor for ice      [kg m^-2 yr^-1 K^-1]
            f_refreeze    (float) refreezing factor               [-]
            **kwargs
        Outputs:
            [A_snow, R, M_melt]  ([theano.tt, theano.tt, theano.tt]) -->
                List of theano tensors without numeric values for each component
                of the mass balance model [m i.e yr^-1]
        """

        pdd_model = TTPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=f_refreeze,
            refreeze_ice=f_refreeze,
        )

        result = pdd_model(temp, precip, std_dev)

        A = result["accu"]
        M = result["melt"]
        R = result["refreeze"]

        return R, A, M


def read_observation(file="DMI-HIRHAM5_1980.nc"):
    """
    Read and return Obs
    """

    with xr.open_dataset(file) as Obs:

        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = stacked.tas.dropna(dim="z").values
        rainfall = stacked.rainfall.dropna(dim="z").values / 1000
        snowfall = stacked.snfall.dropna(dim="z").values / 1000
        smb = stacked.gld.dropna(dim="z").values / 1000
        refreeze = ncl_stacked.rfrz.dropna(dim="z").values / 1000
        melt = stacked.snmel.dropna(dim="z").values / 1000
        precip = rainfall + snowfall

    return (
        temp,
        precip,
        refreeze.sum(axis=0),
        snowfall.sum(axis=0),
        melt.sum(axis=0),
        smb.sum(axis=0),
    )


def plot_posterior(trace, vars: list, out_fp: str):
    """Wrapper for arviz trace plot"""
    _ = az.plot_trace(trace, var_names=vars)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)


def plot_BAMR(z: np.ndarray, ppc: tuple, obs: tuple, out_fp: str, axis_labels=None):
    """Plot net (B)alance, (A)ccumulation, and (M)elt (BAM)."""

    if axis_labels == None:
        axis_labels = (
            "Refreezing (m/yr)",
            "Accumulation (m/yr)",
            "Melt (m/yr)",
            "Refreezing (m/yr)",
        )

    fig, ax = plt.subplots(4, 1, figsize=(5, 7), sharex=True)

    for i, (pp, obs) in enumerate(zip(ppc, obs)):

        ax[i].scatter(z, obs, s=1.0, c="#7570b3")

        ax[i].plot(z, np.mean(pp, axis=(0, 1)), lw=1.0, color="#1b9e77")

        az.plot_hdi(
            z,
            pp,
            hdi_prob=0.6,
            ax=ax[i],
            color="#1b9e77",
            fill_kwargs={"alpha": 0.5, "linewidth": 0},
        )

        az.plot_hdi(
            z,
            pp,
            hdi_prob=0.95,
            ax=ax[i],
            color="#1b9e77",
            fill_kwargs={"alpha": 0.33, "linewidth": 0},
        )

        ax[i].set_ylabel(axis_labels[i])

    ax[0].set_ylim(None, 0.5)
    ax[3].set_xlabel("Elevation [m a.s.l.]")
    plt.tight_layout()
    fig.savefig(out_fp, dpi=300)


def simultaneous_fit_LA(
    T_obs, P_obs, R_obs, A_obs, M_obs, B_obs, draws=4000, tune=2000, cores=1
):
    """Simultaneous fitting the linear accumulation model"""

    const = dict()

    # initialize the PDD melt model class
    PDD_forward = PDD_MCMC(**const)

    # Define Priors
    with pm.Model() as model:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Mass balance Model (physical priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        f_snow_prior = pm.TruncatedNormal("f_snow", mu=4.1, sigma=1.5, lower=0.0)
        f_ice_prior = pm.TruncatedNormal("f_ice", mu=8.0, sigma=2.0, lower=0.0)
        f_refreeze_prior = pm.TruncatedNormal(
            "f_refreeze", mu=0.5, sigma=0.2, lower=0.0, upper=1
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Hyperparameters (likelihood related priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        R_sigma = pm.HalfCauchy("R_sigma", 0.5)
        A_sigma = pm.HalfCauchy("A_sigma", 2)
        M_sigma = pm.HalfCauchy("M_sigma", 5)

    # Define Forward model (wrapped through theano)
    with model:
        R, A, M = PDD_forward.forward(
            T_obs,
            P_obs,
            np.zeros_like(T_obs),
            f_snow_prior,
            f_ice_prior,
            f_refreeze_prior,
        )
        # net balance [m i.e. / yr ]
        B = A + R - M

    # Define likelihood (function?)
    with model:
        # Individual likelihood functions for each component
        R_est = pm.Normal("R_est", mu=R, sigma=R_sigma, observed=R_obs)
        A_est = pm.Normal("A_est", mu=A, sigma=A_sigma, observed=A_obs)
        M_est = pm.Normal("M_est", mu=M, sigma=M_sigma, observed=M_obs)

        potential = pm.Potential("obs", R_est.sum() * A_est.sum() * M_est.sum())

    with model:
        approx = pm.fit(
            draws, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)]
        )

    return approx


def new_likelihood(
    T_obs, P_obs, R_obs, A_obs, M_obs, B_obs, draws=4000, tune=2000, cores=1
):
    """Simultaneous fitting the linear accumulation model"""

    const = dict()

    # initialize the PDD melt model class
    PDD_forward = PDD_MCMC(**const)

    # Define Priors
    with pm.Model() as model:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Mass balance Model (physical priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        f_snow_prior = pm.TruncatedNormal("f_snow", mu=4.1, sigma=1.5, lower=0.0)
        f_ice_prior = pm.TruncatedNormal("f_ice", mu=8.0, sigma=2.0, lower=0.0)
        f_refreeze_prior = pm.TruncatedNormal(
            "f_refreeze", mu=0.5, sigma=0.2, lower=0.0, upper=1
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Hyperparameters (likelihood related priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sigma = pm.HalfCauchy("R_sigma", 1, shape=3)

    # Define Forward model (wrapped through theano)
    with model:
        R, A, M = PDD_forward.forward(
            T_obs,
            P_obs,
            np.zeros_like(T_obs),
            f_snow_prior,
            f_ice_prior,
            f_refreeze_prior,
        )
        # net balance [m i.e. / yr ]
        B = A + R - M

    # Define likelihood (function?)
    with model:
        # Individual likelihood functions for each component
        est = pm.Normal(
            "est",
            mu=np.array([R, A, M]),
            sigma=sigma,
            observed=theano.shared([R_obs, A_obs, M_obs]),
        )

    with model:
        approx = pm.fit(
            draws, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)]
        )

    return approx


def run_LA(fit_type, draws=4000, tune=2000, cores=1):

    if fit_type == "simultaneous":
        simul = True
        new = False
    elif fit_type == "new":
        new = True
        simul = False
    else:
        raise AssertionError("invalid fit type")

    # load observations
    T_obs, P_obs, B_obs, R_obs, A_obs, M_obs, B_obs = read_observation()

    if simul:
        # fit the PDD model with MCMC
        model, trace, pp_R, pp_A, pp_M, pp_B = simultaneous_fit_LA(
            T_obs, P_obs, R_obs, A_obs, M_obs, B_obs, draws, tune, cores
        )
    elif new:
        model, trace, pp_R, pp_A, pp_M, pp_B = new_likelihood(
            T_obs, P_obs, R_obs, A_obs, M_obs, B_obs, draws, tune, cores
        )
    # plot the traces
    vars = ["f_snow", "f_ice", "f_refreeze"]
    out_fp = f"./result/new2/trace_LA_{fit_type}.png"

    plot_posterior(trace, vars=vars, out_fp=out_fp)

    # plot the mass balance results and components
    out_fp = f"./result/new/trace_LA_{fit_type}.png"
    if simul:
        out_fp = f"./result/new/BAM_LA_{fit_type}.png"
        plot_BAMR(
            z_obs, (pp_R, pp_A, pp_M, pp_B), (R_obs, A_obs, M_obs, B_obs), out_fp=out_fp
        )
    elif new:
        out_fp = f"./result/new2/NET_LA_{fit_type}.png"
        plot_NET(z_obs, B_obs, pp_B, out_fp=out_fp)

    # save the trace
    out_fp = f"./result/new2/trace_LA_{fit_type}.nc"
    az.to_netcdf(trace, out_fp)


if __name__ == "__main__":

    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    draws = 1000
    tune = 200
    cores = 6

    # # load observations
    # T_obs, P_obs, R_obs, A_obs, M_obs, B_obs = read_observation()

    n = 100
    m = 12
    rng = np.random.default_rng(2021)
    T_obs = rng.integers(260, 280, (m, n)) + rng.random((m, n))
    P_obs = rng.integers(10, 1000, (m, n)) + rng.random((m, n))
    pdd = PDDModel()
    result = pdd(T_obs, P_obs, np.zeros_like(T_obs) + 4.3)
    B_obs = result["smb"]

    result = pdd(T_obs, P_obs, np.zeros_like(T_obs))
    R_obs = (result["refreeze"] + rng.normal(scale=0.1, size=n)).ravel()
    A_obs = (result["accu"] + rng.normal(scale=1, size=n)).ravel()
    M_obs = (result["melt"] + rng.normal(scale=10, size=n)).ravel()
    B_obs = (result["smb"] + rng.normal(scale=10, size=n)).ravel()

    # initialize the PDD melt model class
    const = dict()
    PDD_forward = PDD_MCMC(**const)

    # Define Priors
    with pm.Model() as model:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Mass balance Model (physical priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        f_snow_prior = pm.TruncatedNormal("f_snow", mu=4.1, sigma=1.5, lower=0.0)
        f_ice_prior = pm.TruncatedNormal("f_ice", mu=8.0, sigma=2.0, lower=0.0)
        f_refreeze_prior = pm.TruncatedNormal(
            "f_refreeze", mu=0.5, sigma=0.2, lower=0.0, upper=1
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----> Hyperparameters (likelihood related priors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        R_sigma = pm.HalfCauchy("R_sigma", 0.5)
        A_sigma = pm.HalfCauchy("A_sigma", 2)
        M_sigma = pm.HalfCauchy("M_sigma", 5)

        sigma = tt.transpose(tt.stack([R_sigma, A_sigma, M_sigma]))

    # Define Forward model (wrapped through theano)
    with model:
        R, A, M = PDD_forward.forward(
            T_obs,
            P_obs,
            np.zeros_like(T_obs),
            f_snow_prior,
            f_ice_prior,
            f_refreeze_prior,
        )
        # net balance [m i.e. / yr ]
        B = A + R - M

        mu = tt.transpose(tt.stack([R, A, M]))

    # Define likelihood (function?)
    with model:
        # Individual likelihood functions for each component
        est = pm.Normal(
            "est",
            mu=mu,
            sigma=sigma,
            observed=np.array(
                [
                    R_obs.reshape(-1, 1),
                    A_obs.reshape(-1, 1),
                    M_obs.reshape(-1, 1),
                ],
            ),
        )

    # with model:
    #     trace = pm.sample(
    #         init="advi",
    #         draws=draws,
    #         tune=tune,
    #         cores=cores,
    #         target_accept=0.9,
    #         return_inferencedata=True,
    #     )

    # az.plot_trace(trace)

    # run inference: Sample
    with model:
        approx = pm.fit(
            draws, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)]
        )

    means = approx.bij.rmap(approx.mean.eval())
    sds = approx.bij.rmap(approx.std.eval())
    import seaborn as sns

    from scipy import stats

    varnames = means.keys()
    fig, axs = plt.subplots(nrows=len(varnames), figsize=(12, 18))
    for var, ax in zip(varnames, axs):
        mu_arr = means[var]
        sigma_arr = sds[var]
        ax.set_title(var)
        for i, (mu, sigma) in enumerate(zip(mu_arr.flatten(), sigma_arr.flatten())):
            sd3 = (-4 * sigma + mu, 4 * sigma + mu)
            x = np.linspace(sd3[0], sd3[1], 300)
            y = stats.norm(mu, sigma).pdf(x)
            ax.plot(x, y)
    fig.tight_layout()
    fig.savefig("test.pdf")
    # trace = pm.sample(
    #     init="adapt_diag",
    #     draws=draws,
    #     tune=tune,
    #     cores=cores,
    #     target_accept=0.9,
    #     scaling=np.power(model.dict_to_array(v_params.stds), 2),
    #     is_cov=True,
    #     return_inferencedata=True,
    # )
