import torch
import numpy as np
import pylab as plt
import xarray as xr
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints


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
        oldy = torch.vstack((array[0], array, array[-1]))
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
        import scipy.special as sp

        # compute positive part of temperature everywhere
        positivepart = torch.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        normtemp = temp / (torch.sqrt(torch.tensor(2)) * stdv)
        calovgreve = stdv / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(
            -(normtemp**2)
        ) + temp / 2 * sp.erfc(-normtemp)

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
        ddf_snow = self.pdd_factor_snow
        ddf_ice = self.pdd_factor_ice

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = torch.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class BayesianLinear(pyro.nn.module.PyroModule):
    def __init__(self, temp, precip, std_dev, B_obs, f_snow, f_ice, f_refreeze):
        super().__init__()
        self.temp = temp
        self.precip = precip
        self.std_dev = std_dev
        self.B_obs = B_obs
        self.f_snow = f_snow
        self.f_ice = f_ice
        self.f_refreeze = f_refreeze

    def forward(self, temp, precip, std_dev):

        f_snow = self.f_snow
        f_ice = self.f_ice
        f_refeeze = self.f_refreeze

        pdd_model = TorchPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=f_refreeze,
            refreeze_ice=f_refreeze,
        )

        result = pdd_model(temp, precip, std_dev)

        A = result["accu"]
        M = result["melt"]
        R = result["refreeze"]
        B = result["smb"]

        # R_sigma = pyro.sample("R_est", dist.HalfCauchy(0.5))
        # A_sigma = pyro.sample("A_est", dist.HalfCauchy(2))
        # M_sigma = pyro.sample("M_est", dist.HalfCauchy(25))
        B_sigma = pyro.sample("B_est", dist.Cauchy(0, 0.5))

        # R_est = pyro.sample("R_est", dist.Normal(R, R_sigma), obs=R_obs)
        # A_est = pyro.sample("A_est", dist.Normal(A, A_sigma), obs=A_obs)
        # M_est = pyro.sample("M_est", dist.Normal(M, M_sigma), obs=M_obs)
        B_est = pyro.sample("B_est", dist.Normal(B, B_sigma), obs=B_obs)

        with pyro.plate("data", x_obs.shape[0]):
            B_est = pyro.sample("B_est", dist.Normal(B, B_sigma), obs=B_obs)


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


if __name__ == "__main__":

    (
        T_obs,
        P_obs,
        R_obs,
        A_obs,
        M_obs,
        B_obs,
    ) = read_observation()

    f_snow = pyro.param("f_snow", lambda: torch.randn(()))
    f_ice = pyro.param("f_ice", lambda: torch.randn(()))
    f_refreeze = pyro.param("f_refreeze", lambda: torch.randn(()))

    # f_snow = pyro.sample("f_snow", dist.Normal(4.1, 1.5))
    # f_ice = pyro.sample("f_ice", dist.Normal(8.0, 2.0))
    # f_refreeze = pyro.sample("f_refreeze", dist.Normal(0.5, 0.2))

    model = BayesianLinear(
        T_obs, P_obs, np.zeros_like(T_obs), B_obs, f_snow, f_ice, f_refreeze
    )
    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())
