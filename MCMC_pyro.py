import logging
import torch
import numpy as np
import pylab as plt
import xarray as xr
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.poutine as poutine

torch.autograd.set_detect_anomaly(True)


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
        pdd_factor_snow=3,
        pdd_factor_ice=8,
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
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class TorchPDDModel(pyro.nn.module.PyroModule):
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
        pdd_factor_snow=3,
        pdd_factor_ice=8,
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
        temp = torch.asarray(temp)
        prec = torch.asarray(prec)
        stdv = torch.asarray(stdv)

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
        snow_depth = torch.zeros_like(temp)
        snow_melt_rate = torch.zeros_like(temp)
        ice_melt_rate = torch.zeros_like(temp)
        snow_refreeze_rate = torch.zeros_like(temp)
        ice_refreeze_rate = torch.zeros_like(temp)

        snow_depth[:-1] = torch.clone(snow_depth[1:])
        snow_depth = snow_depth + accu_rate
        snow_melt_rate, ice_melt_rate = self.melt_rates(snow_depth, inst_pdd)
        snow_depth = snow_depth - snow_melt_rate

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
            res = array * torch.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return torch.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""

        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (torch.arange(len(array) + 2) - 0.5) / len(array)
        oldy = torch.vstack((array[-1], array, array[0]))
        newx = (torch.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)

        return torch.from_numpy(newy)

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
        positivepart = torch.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        normtemp = temp / (torch.sqrt(torch.tensor(2)) * stdv)
        calovgreve = stdv / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(
            -(normtemp**2)
        ) + temp / 2 * torch.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)

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
        snowfrac = torch.clip(reduced_temp, 0, 1)

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
        ddf_snow = self.pdd_factor_snow / 1e3
        ddf_ice = self.pdd_factor_ice / 1e3

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = torch.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class BayesianPDD(pyro.nn.module.PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None):

        with pyro.plate("data", use_cuda=True):

            # f_snow_loc = pyro.param("f_snow_loc", lambda: torch.tensor(4.1))
            # f_snow_scale = pyro.param(
            #     "f_snow_scale",
            #     lambda: torch.tensor(1.5),
            #     constraint=constraints.positive,
            # )
            # f_snow = pyro.sample("f_snow", dist.Normal(f_snow_loc, f_snow_scale))

            # f_ice_loc = pyro.param("f_ice_loc", lambda: torch.tensor(8.0))
            # f_ice_scale = pyro.param(
            #     "f_ice_scale",
            #     lambda: torch.tensor(2.0),
            #     constraint=constraints.positive,
            # )
            # f_ice = pyro.sample("f_ice", dist.Normal(f_ice_loc, f_ice_scale))

            # refreeze_snow_loc = pyro.param(
            #     "refreeze_snow_loc", lambda: torch.tensor(0.5)
            # )
            # refreeze_snow_scale = pyro.param(
            #     "refreeze_snow_scale",
            #     lambda: torch.tensor(0.2),
            #     constraint=constraints.positive,
            # )
            # refreeze_snow = pyro.sample(
            #     "refreeze_snow", dist.Normal(refreeze_snow_loc, refreeze_snow_scale)
            # )
            # refreeze_ice_loc = pyro.param("refreeze_ice_loc", lambda: torch.tensor(0.5))
            # refreeze_ice_scale = pyro.param(
            #     "refreeze_ice_scale",
            #     lambda: torch.tensor(0.2),
            #     constraint=constraints.positive,
            # )
            # refreeze_ice = pyro.sample(
            #     "refreeze_ice", dist.Normal(refreeze_ice_loc, refreeze_ice_scale)
            # )
            f_snow = pyro.sample("f_snow", dist.Normal(4.1, 1.5))
            f_ice = pyro.sample("f_ice", dist.Normal(8.0, 2.0))
            refreeze_snow = pyro.sample("refreeze_snow", dist.Normal(0.5, 0.2))
            refreeze_ice = pyro.sample("refreeze_ice", dist.Normal(0.5, 0.2))

            pdd_model = TorchPDDModel(
                pdd_factor_snow=f_snow,
                pdd_factor_ice=f_ice,
                refreeze_snow=refreeze_snow,
                refreeze_ice=refreeze_ice,
            )

            result = pdd_model(temp, precip, std_dev)

            A = result["accu"]
            M = result["melt"]
            R = result["refreeze"]

            # B_sigma = pyro.sample("B_sigma", dist.HalfCauchy(1))
            # pyro.sample("B_est", dist.Normal(B, B_sigma), obs=B_obs)
            A_sigma = pyro.sample("A_sigma", dist.HalfCauchy(0.5))
            pyro.sample("A_est", dist.Normal(A, A_sigma), obs=A_obs)
            M_sigma = pyro.sample("M_sigma", dist.HalfCauchy(0.5))
            pyro.sample("M_est", dist.Normal(M, M_sigma), obs=M_obs)
            R_sigma = pyro.sample("R_sigma", dist.HalfCauchy(0.5))
            pyro.sample("R_est", dist.Normal(R, R_sigma), obs=R_obs)


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


def model(temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None):

    f_snow = pyro.sample("f_snow", dist.Normal(4.1 / 1, 2.5 / 1))
    f_ice = pyro.sample("f_ice", dist.Normal(8.0 / 1, 2.5 / 1))
    refreeze_snow = pyro.sample("refreeze_snow", dist.Normal(0.5, 0.25))
    refreeze_ice = pyro.sample("refreeze_ice", dist.Normal(0.5, 0.25))

    with pyro.plate("data", use_cuda=True):
        pdd_model = TorchPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=refreeze_snow,
            refreeze_ice=refreeze_ice,
        )

        result = pdd_model(temp, precip, std_dev)
        B = result["smb"]
        A = result["accu"]
        M = result["melt"]
        R = result["refreeze"]

        # B_sigma = pyro.sample("B_sigma", dist.HalfCauchy(1))
        # pyro.sample("B_est", dist.Normal(B, B_sigma), obs=B_obs)
        A_sigma = pyro.sample("A_sigma", dist.HalfCauchy(0.5))
        pyro.sample("A_est", dist.Normal(A, A_sigma), obs=A_obs)
        M_sigma = pyro.sample("M_sigma", dist.HalfCauchy(1))
        pyro.sample("M_est", dist.Normal(M, M_sigma), obs=M_obs)
        R_sigma = pyro.sample("R_sigma", dist.HalfCauchy(1))
        pyro.sample("R_est", dist.Normal(R, R_sigma), obs=R_obs)


def guide(temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None):

    f_snow_loc = pyro.param("f_snow_loc", lambda: torch.tensor(4.1 / 1))
    f_snow_scale = pyro.param(
        "f_snow_scale",
        lambda: torch.tensor(2.5 / 1),
        constraint=constraints.positive,
    )

    f_ice_loc = pyro.param("f_ice_loc", lambda: torch.tensor(8.0 / 1))
    f_ice_scale = pyro.param(
        "f_ice_scale", lambda: torch.tensor(2.5 / 1), constraint=constraints.positive
    )

    refreeze_snow_loc = pyro.param(
        "refreeze_snow_loc",
        lambda: torch.tensor(0.5),
        constraint=constraints.interval(0.0, 1.0),
    )
    refreeze_snow_scale = pyro.param(
        "refreeze_snow_scale",
        lambda: torch.tensor(0.25),
        constraint=constraints.positive,
    )
    refreeze_ice_loc = pyro.param(
        "refreeze_ice_loc",
        lambda: torch.tensor(0.5),
        constraint=constraints.interval(0.0, 1.0),
    )
    refreeze_ice_scale = pyro.param(
        "refreeze_ice_scale",
        lambda: torch.tensor(0.25),
        constraint=constraints.positive,
    )

    with pyro.plate("data", use_cuda=True):
        f_snow = pyro.sample("f_snow", dist.Normal(f_snow_loc, f_snow_scale))
        f_ice = pyro.sample("f_ice", dist.Normal(f_ice_loc, f_ice_scale))
        refreeze_snow = pyro.sample(
            "refreeze_snow", dist.Normal(refreeze_snow_loc, refreeze_snow_scale)
        )
        refreeze_ice = pyro.sample(
            "refreeze_ice", dist.Normal(refreeze_ice_loc, refreeze_ice_scale)
        )

        pdd_model = TorchPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=refreeze_snow,
            refreeze_ice=refreeze_ice,
        )
        result = pdd_model(temp, precip, std_dev)
        B = result["smb"]
        A = result["accu"]
        M = result["melt"]
        R = result["refreeze"]

        # B_sigma = pyro.sample("B_sigma", dist.HalfCauchy(1))
        # pyro.sample(
        #     "B_est",
        #     dist.Normal(B, B_sigma),
        # )
        A_sigma = pyro.sample("A_sigma", dist.HalfCauchy(0.5))
        pyro.sample(
            "A_est",
            dist.Normal(A, A_sigma),
        )

        M_sigma = pyro.sample("M_sigma", dist.HalfCauchy(0.5))
        pyro.sample(
            "M_est",
            dist.Normal(M, M_sigma),
        )

        R_sigma = pyro.sample("R_sigma", dist.HalfCauchy(0.5))
        pyro.sample(
            "R_est",
            dist.Normal(R, R_sigma),
        )
        return {
            "f_snow": f_snow,
            "f_ice": f_ice,
            "refreeze_snow": refreeze_snow,
            "refreeze_ice": refreeze_ice,
        }


if __name__ == "__main__":

    # (
    #     T_obs,
    #     P_obs,
    #     R_obs,
    #     A_obs,
    #     M_obs,
    #     B_obs,
    # ) = read_observation()
    # pyro.clear_param_store()

    n = 100
    m = 12
    rng = np.random.default_rng(2021)
    T_obs = rng.integers(260, 280, (m, n)) + rng.random((m, n))
    P_obs = rng.integers(10, 1000, (m, n)) + rng.random((m, n))
    pdd = PDDModel(
        pdd_factor_snow=4.1, pdd_factor_ice=8, refreeze_snow=0.0, refreeze_ice=0.0
    )
    result = pdd(T_obs, P_obs, np.zeros_like(T_obs))
    B_obs = result["smb"]
    A_obs = result["accu"]
    M_obs = result["melt"]
    R_obs = result["refreeze"]

    tpdd = TorchPDDModel(
        pdd_factor_snow=4.1, pdd_factor_ice=8, refreeze_snow=0.0, refreeze_ice=0.0
    )
    torch_result = tpdd(T_obs, P_obs, np.zeros_like(T_obs))

    # Make sure the two PDD models agree
    # for key in torch_result.keys():
    #     print(
    #         np.testing.assert_array_almost_equal(
    #             torch_result[key], result[key], decimal=2
    #         )
    #     )

    # import sys

    # sys.exit()

    bmodel = BayesianPDD()
    # print("params before:", [name for name, _ in model.named_parameters()])

    print("Making graph")
    graph = pyro.render_model(
        model,
        model_args=(T_obs, P_obs, np.zeros_like(T_obs)),
        filename="model.pdf",
    )
    print("Making graph")
    graph = pyro.render_model(
        bmodel,
        model_args=(T_obs, P_obs, np.zeros_like(T_obs)),
        filename="bmodel.pdf",
    )
    auto_guide = pyro.infer.autoguide.AutoNormal(bmodel)
    adam = pyro.optim.Adam({"lr": 0.01})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    print("Setting up SVI")
    svi = pyro.infer.SVI(model, guide, adam, elbo)

    # pyro.clear_param_store()
    num_iters = 10000
    losses = []
    for i in range(num_iters):
        elbo = svi.step(
            T_obs,
            P_obs,
            np.zeros_like(T_obs),
            torch.from_numpy(A_obs),
            torch.from_numpy(M_obs),
            torch.from_numpy(R_obs),
        )
        losses.append(elbo)
        if i % 100 == 0:
            print(f"Iteration {i} loss: {elbo}")
            logging.info("Elbo loss: {}".format(elbo))
            print(pyro.param("f_snow_loc").item())
            print(pyro.param("f_ice_loc").item())
            print(pyro.param("refreeze_ice_loc").item())
            print(pyro.param("refreeze_snow_loc").item())

    print(pyro.param("f_snow_loc").item())
    print(pyro.param("f_ice_loc").item())
    print(pyro.param("refreeze_ice_loc").item())
    print(pyro.param("refreeze_snow_loc").item())

    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO loss")
    ax.set_ylim(0, 1.0e12)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())

    with pyro.plate("samples", n, dim=-1):
        samples = guide(
            T_obs,
            P_obs,
            np.zeros_like(T_obs),
        )

    fs = samples["f_snow"]
    fi = samples["f_ice"]

    import seaborn as sns

    fig = plt.figure(figsize=(10, 6))

    sns.histplot(fs.detach().cpu().numpy(), kde=True, stat="density", label="f_snow")
    sns.histplot(
        fi.detach().cpu().numpy(),
        kde=True,
        stat="density",
        label="f_ice",
        color="orange",
    )
    plt.legend()
    plt.show()
