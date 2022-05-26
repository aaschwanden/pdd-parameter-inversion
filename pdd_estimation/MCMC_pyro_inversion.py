from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
import numpy as np
import pylab as plt
import xarray as xr
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.poutine as poutine
from scipy.interpolate import interp1d

torch.autograd.set_detect_anomaly(True)


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
        device="cpu",
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
        self.device = device

    def forward(self, temp, prec, stdv=0.0):
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

        device = self.device
        # ensure numpy arrays
        temp = torch.asarray(temp, device=device)
        prec = torch.asarray(prec, device=device)
        stdv = torch.asarray(stdv, device=device)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        if self.interpolate_n >= 1:
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
        oldx = (torch.arange(len(array) + 2, device=self.device) - 0.5) / len(array)
        oldy = torch.vstack((array[-1], array, array[0]))
        newx = (torch.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx.cpu(), oldy.cpu(), kind=rule, axis=0)(newx)

        return torch.from_numpy(newy).to(self.device)

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
    def __init__(self, max_epochs=10000, device="cpu"):
        super().__init__()
        if device == "cuda":
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            use_cuda = True
        else:
            use_cuda = False
        self.use_cuda = use_cuda
        self.device = device
        self.max_epochs = max_epochs

    def model(
        self, temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None, B_obs=None
    ):

        f_snow = pyro.sample("f_snow", dist.Normal(3.0, 1.0)).to(self.device)
        f_ice = pyro.sample("f_ice", dist.Normal(8.0, 1.5)).to(self.device)
        f_refreeze = pyro.sample("f_refreeze", dist.Normal(0.5, 0.2)).to(self.device)

        pdd_model = TorchPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=f_refreeze,
            refreeze_ice=f_refreeze,
            device=self.device,
        )
        result = pdd_model.forward(temp, precip, std_dev)
        A = result["accu"]
        M = result["melt"]
        R = result["refreeze"]

        with pyro.plate("obs", use_cuda=self.use_cuda):

            A_sigma = pyro.sample("A_sigma", dist.Normal(2, 0.2)).to(self.device)
            pyro.sample("A_est", dist.Normal(A, A_sigma).to_event(1), obs=A_obs)
            M_sigma = pyro.sample("M_sigma", dist.Normal(5, 1.0)).to(self.device)
            pyro.sample("M_est", dist.Normal(M, M_sigma).to_event(1), obs=M_obs)
            R_sigma = pyro.sample("R_sigma", dist.Normal(0.5, 0.1)).to(self.device)
            pyro.sample("R_est", dist.Normal(R, R_sigma).to_event(1), obs=R_obs)
            return {
                "f_snow": f_snow,
                "f_ice": f_ice,
                "f_refreeze": f_refreeze,
            }

    def guide(
        self, temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None, B_obs=None
    ):

        f_snow_loc = pyro.param(
            "f_snow_loc",
            torch.tensor(3.0),
            constraint=constraints.interval(1.0, 8.0),
        ).to(self.device)
        f_snow_scale = pyro.param(
            "f_snow_scale",
            torch.tensor(1.0),
            constraint=constraints.positive,
        ).to(self.device)

        f_ice_loc = pyro.param(
            "f_ice_loc",
            torch.tensor(8.0),
            constraint=constraints.interval(1.0, 16.0),
        ).to(self.device)
        f_ice_scale = pyro.param(
            "f_ice_scale", torch.tensor(1.5), constraint=constraints.positive
        ).to(self.device)

        f_refreeze_loc = pyro.param(
            "f_refreeze_loc",
            torch.tensor(0.5),
            constraint=constraints.interval(0.0, 1.0),
        ).to(self.device)
        f_refreeze_scale = pyro.param(
            "f_refreeze_scale",
            lambda: torch.tensor(0.2),
            constraint=constraints.positive,
        ).to(self.device)

        f_snow = pyro.sample("f_snow", dist.Normal(f_snow_loc, f_snow_scale))
        f_ice = pyro.sample("f_ice", dist.Normal(f_ice_loc, f_ice_scale))
        f_refreeze = pyro.sample(
            "f_refreeze", dist.Normal(f_refreeze_loc, f_refreeze_scale)
        )
        pdd_model = TorchPDDModel(
            pdd_factor_snow=f_snow,
            pdd_factor_ice=f_ice,
            refreeze_snow=f_refreeze,
            refreeze_ice=f_refreeze,
            device=self.device,
        )

        result = pdd_model.forward(temp, precip, std_dev)
        return {
            "f_snow": f_snow,
            "f_ice": f_ice,
            "f_refreeze": f_refreeze,
        }

    def forward(
        self, temp, precip, std_dev, A_obs=None, M_obs=None, R_obs=None, B_obs=None
    ):

        pyro.clear_param_store()
        print("Setting up SVI")
        optimizer = torch.optim.Adam
        elbo = pyro.infer.Trace_ELBO()
        scheduler = pyro.optim.ExponentialLR(
            {"optimizer": optimizer, "optim_args": {"lr": 0.1}, "gamma": 0.999}
        )
        svi = pyro.infer.SVI(model.model, model.guide, scheduler, elbo)
        max_epochs = self.max_epochs
        losses = []
        for i in range(max_epochs):
            elbo = svi.step(
                temp,
                precip,
                std_dev,
                A_obs=A_obs,
                M_obs=M_obs,
                R_obs=R_obs,
            )
            scheduler.step()
            losses.append(elbo)
            if i % 1000 == 0:
                print(f"Iteration {i} loss: {elbo}")
                logging.info("Elbo loss: {}".format(elbo))


def read_observation(file="../data/DMI-HIRHAM5_1980_MM.nc", thinning_factor=1):
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
        temp[..., ::thinning_factor],
        precip[..., ::thinning_factor],
        refreeze.sum(axis=0)[..., ::thinning_factor],
        snowfall.sum(axis=0)[..., ::thinning_factor],
        melt.sum(axis=0)[..., ::thinning_factor],
        smb.sum(axis=0)[..., ::thinning_factor],
    )


def load_synth_climate(f_snow=3, f_ice=8, f_refreeze=0, device="cpu"):
    n = 10
    m = 12

    lx = ly = 750000
    x = np.linspace(-lx, lx, n)
    y = np.linspace(-ly, ly, n)
    t = (np.arange(12) + 0.5) / 12

    # assign temperature and precipitation values
    (yy, xx) = np.meshgrid(y, x)
    temp = np.zeros((m, n, n))
    prec = np.zeros((m, n, n))
    stdv = np.zeros((m, n, n))
    for i in range(len(t)):
        temp[i] = -10 * yy / ly - 5 * np.cos(i * 2 * np.pi / 12)
        prec[i] = xx / lx * (np.sign(xx) - np.cos(i * 2 * np.pi / 12))
        stdv[i] = (2 + xx / lx - yy / ly) * (1 - np.cos(i * 2 * np.pi / 12))

    T_obs = temp.reshape(m, -1)
    P_obs = prec.reshape(m, -1)
    std_dev = stdv.reshape(m, -1)
    pdd = TorchPDDModel(
        pdd_factor_snow=f_snow,
        pdd_factor_ice=f_ice,
        refreeze_snow=f_refreeze,
        refreeze_ice=f_refreeze,
        device=device,
    )
    std_dev = np.zeros_like(T_obs)
    result = pdd(T_obs, P_obs, std_dev)

    A_obs = result["accu"]
    M_obs = result["melt"]
    R_obs = result["refreeze"]

    return (
        torch.from_numpy(T_obs).to(device),
        torch.from_numpy(P_obs).to(device),
        torch.from_numpy(std_dev).to(device),
        A_obs,
        M_obs,
        R_obs,
    )


def load_hirham_climate(f_snow=3, f_ice=8, f_refreeze=0, device="cpu"):
    (
        T_obs,
        P_obs,
        R_obs,
        A_obs,
        M_obs,
        B_obs,
    ) = read_observation(thinning_factor=100)

    T_obs -= 273.15
    pdd = TorchPDDModel(
        pdd_factor_snow=f_snow,
        pdd_factor_ice=f_ice,
        refreeze_snow=f_refreeze,
        refreeze_ice=f_refreeze,
        devic=device,
    )
    std_dev = np.zeros_like(T_obs)
    result = pdd(T_obs, P_obs, std_dev)
    # A_obs = result["accu"]
    # M_obs = result["melt"]
    # R_obs = result["refreeze"]

    return (
        torch.from_numpy(T_obs).to(device),
        torch.from_numpy(P_obs).to(device),
        torch.from_numpy(std_dev).to(device),
        torch.from_numpy(A_obs).to(device),
        torch.from_numpy(M_obs).to(device),
        torch.from_numpy(R_obs).to(device),
    )


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Variational Inference of PDD parameters."
    parser.add_argument(
        "-c",
        "--climate",
        dest="climate",
        choices=["hirham", "synth"],
        help="Climate (temp, precip) forcing",
        default="synth",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device (cpu, cuda)",
        default="cpu",
    )
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="Number of iterations (epochs).",
        type=int,
        default=10000,
    )
    options = parser.parse_args()
    climate = options.climate
    device = options.device
    max_epochs = options.max_epochs
    pyro.clear_param_store()

    fs = 4
    fi = 8
    fr = 0.0

    print("-------------------------------------------")
    print("Variantional Inference of PDD parameters")
    print("-------------------------------------------\n")
    print("Trying to recover:")
    print(f"f_snow={fs}, f_ice={fi}, f_refreeze={fr}\n")
    print("-------------------------------------------")

    if climate == "synth":
        T_obs, P_obs, std_dev, A_obs, M_obs, R_obs = load_synth_climate(
            f_snow=fs, f_ice=fi, f_refreeze=fr, device=device
        )
    elif climate == "hirham":
        T_obs, P_obs, std_dev, A_obs, M_obs, R_obs = load_hirham_climate(
            f_snow=fs, f_ice=fi, f_refreeze=fr, device=device
        )
    else:
        print(f"Climate {climate} not recognized")

    # Normalization does not seem to improve convergence
    T_obs_norm = (T_obs - T_obs.mean(axis=1).reshape(-1, 1)) / T_obs.std(
        axis=1
    ).reshape(-1, 1)
    P_obs_norm = (P_obs - P_obs.mean(axis=1).reshape(-1, 1)) / P_obs.std(
        axis=1
    ).reshape(-1, 1)
    std_dev_norm = torch.nan_to_num(
        (std_dev - std_dev.mean(axis=1).reshape(-1, 1))
        / std_dev.std(axis=1).reshape(-1, 1)
    )

    model = BayesianPDD(device=device, max_epochs=max_epochs)
    model.forward(T_obs, P_obs, std_dev, A_obs=A_obs, M_obs=M_obs, R_obs=R_obs)
    print("Recovered parameters")
    for name, value in pyro.get_param_store().items():
        print(name, f"""{pyro.param(name).data.cpu().numpy():.2f}""")

    with pyro.plate("samples", T_obs.shape[1], dim=-1):
        samples = model.guide(
            T_obs,
            P_obs,
            std_dev,
        )

    f_snow_posterior = samples["f_snow"].detach().cpu().numpy()
    f_ice_posterior = samples["f_ice"].detach().cpu().numpy()
    f_refreeze_posterior = samples["f_refreeze"].detach().cpu().numpy()

    import seaborn as sns
    from scipy.stats import norm

    fig, axs = plt.subplots(
        1,
        2,
        figsize=[12.0, 4],
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    sns.histplot(
        f_snow_posterior,
        kde=True,
        stat="density",
        label="f_snow_posterior",
        lw=0,
        ax=axs[0],
    )
    sns.histplot(
        f_ice_posterior,
        kde=True,
        stat="density",
        label="f_ice_posterior",
        lw=0,
        color="orange",
        ax=axs[0],
    )
    sns.histplot(
        f_refreeze_posterior,
        kde=True,
        stat="density",
        lw=0,
        label="f_refreeze_posterior",
        ax=axs[1],
    )
    x0 = np.arange(0, 16, 0.001)
    x1 = np.arange(0, 1, 0.001)
    axs[0].plot(x0, norm.pdf(x0, 3, 1.0), lw=2, ls="dotted", label="f_snow_prior")
    axs[0].plot(
        x0,
        norm.pdf(x0, 8.0, 1.5),
        color="orange",
        lw=2,
        ls="dotted",
        label="f_ice_prior",
    )
    axs[1].plot(x1, norm.pdf(x1, 0.5, 0.2), lw=2, ls="dotted", label="f_refreeze_prior")

    if climate != "hirham":
        axs[0].axvline(fs, lw=3, label="f_snow_true")
        axs[0].axvline(fi, lw=3, color="orange", label="f_ice_true")
        axs[1].axvline(fr, lw=3, label="f_refreeze_true")
    axs[0].legend()
    axs[1].legend()
    fig.savefig(f"climate_{climate}_snow_{fs}_ice_{fi}_refreeze_{fr}.pdf")
