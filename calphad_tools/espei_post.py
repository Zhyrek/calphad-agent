"""Post-processing of ESPEI MCMC results and propagation of the parameter uncertainty.

Registered by :func:`register` (called from ``calphad_tools.agent.tools``):

* ``analyze_mcmc``          - convergence diagnostics + posterior statistics of an ESPEI run
  (``trace.npy`` / ``lnprob.npy``): zero-filled (unfinished) iterations removed, automatic burn-in,
  per-parameter mean/median/std/95 % credible interval, acceptance fraction, integrated
  autocorrelation time + effective sample size, split-R-hat; lnprob, trace, corner and
  correlation-matrix plots; every fitted symbol (VV0000...) is mapped to the TDB parameter(s)
  that use it.
* ``mcmc_to_tdb``           - write a TDB with the posterior mean / median / MAP parameter set.
* ``propagate_uncertainty`` - draw posterior samples and recompute a binary phase diagram (all
  sample boundaries + the posterior-median diagram, invariant-temperature distributions) or an
  equilibrium property along one condition axis (median + 95 % band).

How ESPEI stores the chains (verified in espei 0.9.1 / emcee 2.2.1):
``trace.npy`` = ``sampler.chain`` with shape (chains, iterations, parameters) and ``lnprob.npy`` =
``sampler.lnprobability`` (chains, iterations); both are pre-allocated for all iterations and saved
every ``save_interval`` steps, so an unfinished run is zero-filled at the end
(``espei.analysis.truncate_arrays`` removes those).  The parameter columns follow
``setup_context``: ``sorted(symbols)`` where ``symbols`` is the ``mcmc.symbols`` YAML setting or,
by default, ``espei.utils.database_symbols_to_fit(dbf)`` = sorted names of the database symbols
matching ``^V[V]?([0-9]+)$``.

Everything is plain functions so it can also be used without the agent.
"""
from __future__ import annotations

import copy
import csv
import json
import re
import tempfile
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullLocator  # noqa: E402

KELVIN = 273.15
MAX_SAMPLES = 200
DEFAULT_SYMBOL_REGEX = r"^V[V]?([0-9]+)$"
DPI = 150


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def truncate_zero_filled(trace: np.ndarray, lnprob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Remove the zero-filled iterations of an unfinished ESPEI run.

    Uses ``espei.analysis.truncate_arrays`` (iterations counted from the non-zero parameter rows);
    if that fails (e.g. a parameter that is exactly zero in only some chains makes its reshape
    impossible) the last iteration in which any chain has a non-zero parameter vector is used."""
    how = "espei.analysis.truncate_arrays"
    try:
        from espei.analysis import truncate_arrays
        tr, lp = truncate_arrays(trace, lnprob)
    except Exception:  # noqa: BLE001 - fall back to the equivalent robust rule
        how = "last iteration with any non-zero parameter vector"
        done = np.any(trace != 0, axis=(0, 2))
        n = int(np.nonzero(done)[0][-1] + 1) if done.any() else 0
        tr, lp = trace[:, :n, :], lnprob[:, :n]
    # an iteration with all-zero parameters but non-zero lnprob would be real data: never cut it
    nz_lp = np.nonzero(np.any(lnprob != 0, axis=0))[0]
    if nz_lp.size and nz_lp[-1] + 1 > tr.shape[1]:
        n = int(nz_lp[-1] + 1)
        tr, lp, how = trace[:, :n, :], lnprob[:, :n], how + " (extended to the last non-zero lnprob)"
    return tr, lp, how


def load_chains(trace_path: str, lnprob_path: str) -> Dict:
    trace = np.load(trace_path)
    lnprob = np.load(lnprob_path)
    if trace.ndim != 3:
        raise ValueError(f"{trace_path}: expected a (chains, iterations, parameters) array, got shape {trace.shape}")
    if lnprob.shape != trace.shape[:2]:
        raise ValueError(f"lnprob shape {lnprob.shape} does not match trace shape {trace.shape} "
                         "(expected (chains, iterations))")
    tr, lp, how = truncate_zero_filled(trace, lnprob)
    if tr.shape[1] == 0:
        raise ValueError("the trace contains no completed iterations (all zeros)")
    lp = np.where(np.isfinite(lp), lp, np.nan)
    return {"trace": tr.astype(float), "lnprob": lp.astype(float), "allocated_iterations": int(trace.shape[1]),
            "completed_iterations": int(tr.shape[1]), "truncation": how}


def resolve_run_files(trace: str, lnprob: Optional[str] = None, tdb: Optional[str] = None,
                      need_tdb: bool = True) -> Tuple[str, str, Optional[str]]:
    """Accept a run directory or explicit paths; missing lnprob / tdb default to siblings of the trace
    (lnprob.npy; mcmc.tdb, else dft.tdb, else the only *.tdb)."""
    tp = Path(trace)
    if tp.is_dir():
        d = tp
        tp = d / "trace.npy"
    else:
        d = tp.parent
    if not tp.exists():
        raise FileNotFoundError(f"trace file {tp} not found")
    lpp = Path(lnprob) if lnprob else d / "lnprob.npy"
    if not lpp.exists():
        raise FileNotFoundError(f"lnprob file {lpp} not found (give `lnprob`)")
    tdb_p = None
    if tdb:
        tdb_p = tdb
    elif need_tdb:
        for cand in ("mcmc.tdb", "dft.tdb"):
            if (d / cand).exists():
                tdb_p = str(d / cand)
                break
        else:
            tdbs = sorted(d.glob("*.tdb"))
            if len(tdbs) == 1:
                tdb_p = str(tdbs[0])
            else:
                raise FileNotFoundError("give `tdb` (the MCMC input or output database with the VV symbols)")
    return str(tp), str(lpp), tdb_p


def _load_db(tdb: str):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(tdb)


def _unpack(x) -> float:
    try:
        from espei.utils import unpack_piecewise
        return float(unpack_piecewise(x))
    except Exception:  # noqa: BLE001
        import symengine
        if isinstance(x, symengine.Piecewise):
            return float(x.args[0])
        return float(x)


def fitted_symbols(db, n_params: int, symbols: Optional[Sequence[str]] = None) -> List[str]:
    """Column names of the trace, exactly as ESPEI orders them (see module docstring)."""
    if symbols:
        syms = sorted(str(s) for s in symbols)
        missing = [s for s in syms if s not in db.symbols]
        if missing:
            raise ValueError(f"symbols {missing} are not defined in the database")
    else:
        try:
            from espei.utils import database_symbols_to_fit
            syms = list(database_symbols_to_fit(db))
        except Exception:  # noqa: BLE001
            pat = re.compile(DEFAULT_SYMBOL_REGEX)
            syms = sorted(s for s in db.symbols if pat.match(s))
    if len(syms) != n_params:
        raise ValueError(f"the trace has {n_params} parameter columns but the database has {len(syms)} fitting "
                         f"symbols {syms[:12]}{'...' if len(syms) > 12 else ''}. Use the TDB that was the MCMC "
                         "input (or its output), or give `symbols` = the mcmc.symbols list of the ESPEI YAML.")
    return syms


def _role(d) -> str:
    s = str(d).replace(" ", "")
    return {"1": "const", "T": "T", "T*log(T)": "T*LN(T)", "log(T)*T": "T*LN(T)", "T**2": "T**2",
            "T**3": "T**3", "T**(-1)": "1/T", "1/T": "1/T"}.get(s, s)


def symbol_usage(db, symbols: Sequence[str]) -> Dict[str, Dict]:
    """For every symbol: TDB parameters (and FUNCTIONs) whose expression contains it, and its role
    (coefficient of 1, T, T*LN(T), ...), plus the value stored in the database."""
    import symengine
    from .mapping_tools import _param_str
    out = {s: {"parameters": [], "roles": [], "value_in_tdb": None, "via_functions": []} for s in symbols}
    for s in symbols:
        try:
            out[s]["value_in_tdb"] = _unpack(db.symbols[s])
        except Exception:  # noqa: BLE001
            pass
    sym_objs = {s: symengine.Symbol(s) for s in symbols}

    def pieces(expr):
        if isinstance(expr, symengine.Piecewise):
            return list(expr.args[0::2])
        return [expr]

    def uses(expr, s):
        return any(sym_objs[s] in p.free_symbols for p in pieces(expr))

    def role(expr, s):
        roles = set()
        for p in pieces(expr):
            if sym_objs[s] in p.free_symbols:
                roles.add(_role(symengine.diff(p, sym_objs[s])))
        return ",".join(sorted(roles))

    # functions (other than the fitted symbols) that reference a fitted symbol
    carrier: Dict[str, List[str]] = {}
    for fname, fexpr in db.symbols.items():
        if fname in sym_objs:
            continue
        try:
            for s in symbols:
                if uses(fexpr, s):
                    carrier.setdefault(fname, []).append(s)
                    out[s]["via_functions"].append(f"{fname} ({role(fexpr, s)})")
        except Exception:  # noqa: BLE001
            continue
    for p in db._parameters.all():
        expr = p["parameter"]
        try:
            free = set().union(*[q.free_symbols for q in pieces(expr)])
        except Exception:  # noqa: BLE001
            continue
        names = {str(f) for f in free}
        label = _param_str(p)
        for s in symbols:
            if s in names:
                out[s]["parameters"].append(label)
                out[s]["roles"].append(role(expr, s))
        for fname, syms in carrier.items():
            if fname in names:
                for s in syms:
                    out[s]["parameters"].append(f"{label} via {fname}")
                    out[s]["roles"].append("indirect")
    return out


def _describe(usage: Dict, s: str) -> str:
    u = usage.get(s) or {}
    items = [f"{p} [{r}]" for p, r in zip(u.get("parameters", []), u.get("roles", []))]
    return "; ".join(items) if items else "(not used by any parameter)"


def _short(usage: Dict, s: str) -> str:
    u = usage.get(s) or {}
    if not u.get("parameters"):
        return ""
    p, r = u["parameters"][0], u["roles"][0]
    more = f" +{len(u['parameters']) - 1}" if len(u["parameters"]) > 1 else ""
    return f"{p} {'' if r == 'const' else '· ' + r}{more}".strip()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def estimate_burn_in(lnprob: np.ndarray, k: float = 3.0) -> Tuple[int, Dict]:
    """Burn-in from lnprob stabilisation.

    The reference level is the median lnprob of the second half of the run; its noise is the larger
    of the robust (MAD) scatter of successive per-iteration medians over the second half and the
    standard error of a per-iteration median over chains (both insensitive to a remaining drift).  The per-iteration median (lightly smoothed) must be
    within ``k`` noise units of the reference: burn-in ends after the last iteration in the first 75 %
    of the run that is below that band."""
    lp = np.asarray(lnprob, float)
    n = lp.shape[1]
    info: Dict = {"method": "lnprob stabilisation", "k": k}
    if n < 8:
        info["note"] = "too few iterations for an automatic estimate; burn-in = 0"
        return 0, info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(lp, axis=0)
        med = np.where(np.isfinite(med), med, np.nanmin(lp[np.isfinite(lp)]) if np.isfinite(lp).any() else 0.0)
        w = max(1, n // 50)
        med_s = np.convolve(np.pad(med, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
        half = lp[:, n // 2:]
        ref = float(np.nanmedian(half))
        m2 = med[n // 2:]
        # noise of the per-iteration median, insensitive to a remaining drift: from successive differences
        spread_iter = 1.4826 * float(np.nanmedian(np.abs(np.diff(m2)))) / np.sqrt(2.0) if len(m2) > 2 else 0.0
        # spread across chains at fixed iteration (also drift-free) -> standard error of the median
        spread_chain = 1.4826 * float(np.nanmedian(np.abs(half - np.nanmedian(half, axis=0))))
        n_ch = max(1, int(np.sum(np.isfinite(lp[:, -1]))))
        sem = 1.2533 * spread_chain / np.sqrt(n_ch)
    tol = max(k * max(spread_iter, sem), 1e-9 * abs(ref), 1e-12)
    ok = med_s >= ref - tol
    lim = int(np.ceil(0.75 * n))
    bad = np.nonzero(~ok[:lim])[0]
    burn = int(bad[-1] + 1) if bad.size else 0
    info.update({"reference_lnprob": ref, "tolerance": tol, "smoothing_window": w, "burn_in": burn})
    late_bad = int(np.sum(~ok[lim:]))
    if burn > 0.5 * n:
        info["warning"] = ("lnprob was still rising during the first half of the run (burn-in > 50 % of the "
                           "iterations): the chains have probably not converged - run longer")
    elif late_bad > 0.1 * (n - lim):
        info["warning"] = "lnprob median drops below the stable level repeatedly near the end of the run"
    return burn, info


def acceptance_fraction(trace: np.ndarray) -> np.ndarray:
    """Per chain: fraction of steps in which the parameter vector changed (emcee accepts a proposal
    as a whole; a rejected move repeats the previous position exactly)."""
    if trace.shape[1] < 2:
        return np.full(trace.shape[0], np.nan)
    moved = np.any(np.diff(trace, axis=1) != 0, axis=2)
    return moved.mean(axis=1)


def _acf_1d(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT (as emcee>=3 ``autocorr.function_1d``; emcee 2.2.1's
    ``autocorr.function`` is broken with numpy 2 - list indexing)."""
    x = np.asarray(x, float)
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x - x.mean(), n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    return acf / acf[0]


def integrated_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of one parameter; x has shape (steps, walkers).  The ACF is
    averaged over walkers (constant, i.e. stuck, walkers are skipped) and Sokal's automatic window
    (smallest M with M >= c*tau(M)) is used, as in emcee's ``integrated_time``."""
    try:
        from emcee.autocorr import function_1d  # emcee >= 3
    except ImportError:
        function_1d = _acf_1d
    good = [j for j in range(x.shape[1]) if np.ptp(x[:, j]) > 0]
    if not good or x.shape[0] < 2:
        return float("nan")
    f = np.mean([function_1d(x[:, j]) for j in good], axis=0)
    taus = 2.0 * np.cumsum(f) - 1.0
    m = np.arange(len(taus)) < c * taus
    window = int(np.argmin(m)) if np.any(~m) else len(taus) - 1
    return float(max(taus[window], 1.0))


def split_rhat(x: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction on split chains; x has shape (walkers, steps).  NB: the
    walkers of an affine-invariant ensemble are not independent, so this is indicative only."""
    n = x.shape[1] // 2
    if n < 2:
        return float("nan")
    ch = np.concatenate([x[:, :n], x[:, x.shape[1] - n:]], axis=0)
    m = ch.mean(axis=1)
    W = float(np.mean(ch.var(axis=1, ddof=1)))
    B = float(n * np.var(m, ddof=1))
    if W <= 0:
        return float("nan") if B <= 0 else float("inf")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def _burn(burn_in, lnprob) -> Tuple[int, Dict]:
    n = lnprob.shape[1]
    if burn_in is None or (isinstance(burn_in, str) and burn_in.strip().lower() == "auto"):
        return estimate_burn_in(lnprob)
    b = int(burn_in)
    if b < 0 or b >= n:
        raise ValueError(f"burn_in must be between 0 and {n - 1} (completed iterations: {n})")
    return b, {"method": "user", "burn_in": b}


def posterior_samples(trace: np.ndarray, burn: int, thin: int = 1) -> np.ndarray:
    thin = max(1, int(thin or 1))
    return trace[:, burn::thin, :].reshape(-1, trace.shape[2])


def summarize_chains(trace: np.ndarray, lnprob: np.ndarray, symbols: Sequence[str], burn: int, thin: int = 1,
                     tau_tol: float = 50.0) -> Dict:
    """Posterior statistics and convergence diagnostics (post burn-in)."""
    post_tr = trace[:, burn:, :]
    thin = max(1, int(thin or 1))
    samples = posterior_samples(trace, burn, thin)
    n_steps = post_tr.shape[1]
    acc = acceptance_fraction(post_tr)
    flat_lp = lnprob.reshape(-1)
    i_map = int(np.nanargmax(flat_lp)) if np.isfinite(flat_lp).any() else 0
    c_map, it_map = divmod(i_map, lnprob.shape[1])
    params = []
    warns = []
    for k, s in enumerate(symbols):
        x = samples[:, k]
        q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        chains_k = post_tr[:, :, k]
        tau = integrated_time(chains_k.T) if n_steps >= 4 else float("nan")
        n_good = int(sum(np.ptp(chains_k[j]) > 0 for j in range(chains_k.shape[0])))
        ess = float(min(n_steps * n_good / tau, n_steps * chains_k.shape[0])) if np.isfinite(tau) else float("nan")
        params.append({
            "symbol": s, "mean": float(np.mean(x)), "median": float(q[2]), "std": float(np.std(x, ddof=1))
            if len(x) > 1 else 0.0, "ci95": [float(q[0]), float(q[4])], "ci68": [float(q[1]), float(q[3])],
            "map": float(trace[c_map, it_map, k]), "tau": tau,
            "tau_reliable": bool(np.isfinite(tau) and n_steps > tau_tol * tau), "ess": ess,
            "rhat": split_rhat(chains_k)})
    unreliable = [p["symbol"] for p in params if not p["tau_reliable"]]
    if unreliable:
        warns.append(f"autocorrelation time not reliable for {len(unreliable)}/{len(params)} parameters "
                     f"(post-burn-in chain of {n_steps} steps is shorter than {tau_tol:g} tau): run longer")
    bad_r = [p["symbol"] for p in params if np.isfinite(p["rhat"]) and p["rhat"] > 1.1]
    if bad_r:
        warns.append(f"split R-hat > 1.1 for {', '.join(bad_r[:10])}{'...' if len(bad_r) > 10 else ''} "
                     "(chains disagree; not converged)")
    acc_m = float(np.nanmean(acc)) if np.isfinite(acc).any() else float("nan")
    stuck = int(np.sum(acc == 0))
    if np.isfinite(acc_m) and acc_m < 0.1:
        warns.append(f"low mean acceptance fraction {acc_m:.3f} (emcee typically 0.2-0.5)")
    if stuck:
        warns.append(f"{stuck} chain(s) never moved after burn-in")
    corr = _corr(samples)
    pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.8:
                pairs.append({"pair": [symbols[i], symbols[j]], "r": round(float(corr[i, j]), 3)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    return {"n_chains": int(trace.shape[0]), "n_parameters": int(trace.shape[2]), "burn_in": int(burn),
            "thin": thin, "post_burn_in_steps": int(n_steps), "n_samples": int(samples.shape[0]),
            "acceptance_fraction": {"mean": acc_m, "min": float(np.nanmin(acc)) if acc.size else float("nan"),
                                    "max": float(np.nanmax(acc)) if acc.size else float("nan"),
                                    "stuck_chains": stuck},
            "map": {"lnprob": float(flat_lp[i_map]), "chain": int(c_map), "iteration": int(it_map)},
            "lnprob_final_median": float(np.nanmedian(lnprob[:, -1])),
            "parameters": params, "correlation": corr, "strong_correlations": pairs, "warnings": warns}


def _corr(samples: np.ndarray) -> np.ndarray:
    k = samples.shape[1]
    sd = samples.std(axis=0)
    c = np.full((k, k), np.nan)
    ok = sd > 0
    if ok.sum() >= 1 and samples.shape[0] > 1:
        z = (samples[:, ok] - samples[:, ok].mean(axis=0)) / sd[ok]
        sub = (z.T @ z) / samples.shape[0]
        idx = np.nonzero(ok)[0]
        c[np.ix_(idx, idx)] = np.clip(sub, -1, 1)
    return c


# ---------------------------------------------------------------------------
# plots (plain matplotlib, picklable: no lambdas on artists / formatters)
# ---------------------------------------------------------------------------
def _grid(n: int, max_cols: int = 4) -> Tuple[int, int]:
    cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n)))))
    return int(np.ceil(n / cols)), cols


def plot_lnprob(lnprob: np.ndarray, burn: int, out_png: str, title: str = "") -> str:
    n_ch, n = lnprob.shape
    it = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    alpha = float(np.clip(8.0 / n_ch, 0.08, 0.5))
    for c in range(n_ch):
        ax.plot(it, lnprob[c], color="0.55", lw=0.6, alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(lnprob, axis=0)
    ax.plot(it, med, color="#d62728", lw=1.8, label="median over chains")
    ax.plot([], [], color="0.55", lw=0.8, label=f"{n_ch} chains")
    if burn > 0:
        ax.axvspan(0, burn, color="0.85", alpha=0.6, lw=0, label=f"burn-in ({burn})")
    post = lnprob[:, burn:]
    fin = post[np.isfinite(post)]
    if fin.size:
        lo, hi = np.percentile(fin, [1, 100])
        allfin = lnprob[np.isfinite(lnprob)]
        span = max(hi - lo, 1e-9)
        if allfin.min() < lo - 3 * span:           # early lnprob would squash the plot: clip
            ax.set_ylim(lo - 1.5 * span, hi + 0.15 * span)
            ax.text(0.99, 0.02, "y axis clipped (early iterations lower)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color="0.4")
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ln(probability)")
    ax.set_title(title or "MCMC log-probability", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right" if fin.size else "best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_traces(trace: np.ndarray, symbols: Sequence[str], burn: int, out_png: str, usage: Dict,
                medians: Sequence[float]) -> str:
    n_ch, n, k = trace.shape
    rows, cols = _grid(k)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.2 * rows), squeeze=False, sharex=True)
    alpha = float(np.clip(6.0 / n_ch, 0.06, 0.5))
    it = np.arange(n)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= k:
            ax.set_visible(False)
            continue
        for c in range(n_ch):
            ax.plot(it, trace[c, :, idx], color="#1f77b4", lw=0.5, alpha=alpha)
        if burn > 0:
            ax.axvspan(0, burn, color="0.85", alpha=0.6, lw=0)
        ax.axhline(medians[idx], color="#d62728", lw=1.0, ls="--")
        ax.set_title(f"{symbols[idx]}\n{_short(usage, symbols[idx])}", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.get_offset_text().set_fontsize(6)
        if idx // cols == rows - 1 or idx + cols >= k:
            ax.set_xlabel("Iteration", fontsize=8)
            ax.tick_params(labelbottom=True)
    fig.suptitle(f"Parameter traces ({n_ch} chains; grey = burn-in, dashed = posterior median)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_corner(samples: np.ndarray, symbols: Sequence[str], out_png: str, bins: int = 25) -> str:
    """Corner plot: 1-D marginals (median, 95 % CI) on the diagonal, pairwise 2-D histograms (or
    scatter for few samples) below it."""
    k = samples.shape[1]
    size = max(5.0, min(1.75 * k, 22.0))
    fig, axes = plt.subplots(k, k, figsize=(size, size), squeeze=False)
    q = np.percentile(samples, [2.5, 50, 97.5], axis=0)
    few = samples.shape[0] < 400
    lims = []
    for j in range(k):
        lo, hi = samples[:, j].min(), samples[:, j].max()
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(lo) * 1e-6, 1e-12)
        lims.append((lo - pad, hi + pad))
    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(samples[:, j], bins=bins, range=lims[j], histtype="stepfilled", color="#9ecae1",
                        edgecolor="#3182bd", lw=0.8)
                ax.axvline(q[1, j], color="#d62728", lw=1.0)
                ax.axvline(q[0, j], color="#d62728", lw=0.8, ls="--")
                ax.axvline(q[2, j], color="#d62728", lw=0.8, ls="--")
                ax.set_title(f"{symbols[j]}\n{q[1, j]:.4g}\n[{q[0, j]:.3g}, {q[2, j]:.3g}]", fontsize=6.5)
            else:
                if few:
                    ax.scatter(samples[:, j], samples[:, i], s=3, color="#08519c", alpha=0.35, lw=0)
                else:
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, range=[lims[j], lims[i]], cmap="Blues",
                              cmin=1)
                ax.axvline(q[1, j], color="#d62728", lw=0.6)
                ax.axhline(q[1, i], color="#d62728", lw=0.6)
                ax.set_ylim(*lims[i])
            ax.set_xlim(*lims[j])
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(NullLocator() if i == j else MaxNLocator(3, prune="both"))
            ax.tick_params(labelsize=6)
            ax.xaxis.get_offset_text().set_fontsize(6)
            ax.yaxis.get_offset_text().set_fontsize(6)
            if i == k - 1:
                ax.set_xlabel(symbols[j], fontsize=8)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(symbols[i], fontsize=8)
            elif i != j:
                ax.tick_params(labelleft=False)
    # summary in the empty upper triangle (a suptitle would float far above a tall grid)
    fig.text(0.97, 0.97, f"Posterior ({samples.shape[0]} samples)\ndiagonal: marginal, median (solid)\n"
             "and 95 % credible interval (dashed)", ha="right", va="top", fontsize=10 if k > 3 else 8)
    fig.subplots_adjust(wspace=0.08, hspace=0.08, top=0.93 if k < 4 else 0.96)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_correlation(corr: np.ndarray, symbols: Sequence[str], out_png: str) -> str:
    k = len(symbols)
    size = max(4.5, min(0.55 * k + 2.5, 16))
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    im = ax.imshow(np.ma.masked_invalid(corr), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(symbols, rotation=90, fontsize=7)
    ax.set_yticklabels(symbols, fontsize=7)
    if k <= 16:
        for i in range(k):
            for j in range(k):
                if np.isfinite(corr[i, j]):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(corr[i, j]) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson correlation")
    ax.set_title("Posterior parameter correlation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan" if x is not None else ""
    return f"{x:.6g}"


def analyze_mcmc(trace: str, lnprob: str, tdb: str, out_prefix: str, *, burn_in: Union[int, str, None] = "auto",
                 thin: int = 1, symbols: Optional[Sequence[str]] = None,
                 corner_parameters: Optional[Sequence[str]] = None, max_corner: int = 12) -> Dict:
    ch = load_chains(trace, lnprob)
    tr, lp = ch["trace"], ch["lnprob"]
    db = _load_db(tdb)
    syms = fitted_symbols(db, tr.shape[2], symbols)
    usage = symbol_usage(db, syms)
    burn, binfo = _burn(burn_in, lp)
    st = summarize_chains(tr, lp, syms, burn, thin)
    if binfo.get("warning"):
        st["warnings"].insert(0, binfo["warning"])
    for p in st["parameters"]:
        u = usage[p["symbol"]]
        p["tdb_parameters"] = [f"{a} [{r}]" for a, r in zip(u["parameters"], u["roles"])]
        p["value_in_tdb"] = u["value_in_tdb"]
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    files = {}
    files["lnprob_png"] = plot_lnprob(lp, burn, out_prefix + "_lnprob.png",
                                      title=f"MCMC log-probability ({Path(trace).parent.name or Path(trace).stem})")
    files["traces_png"] = plot_traces(tr, syms, burn, out_prefix + "_traces.png", usage,
                                      [p["median"] for p in st["parameters"]])
    samples = posterior_samples(tr, burn, thin)
    if corner_parameters:
        cp = [s.upper() for s in corner_parameters]
        bad = [s for s in cp if s not in syms]
        if bad:
            raise ValueError(f"corner_parameters {bad} are not fitted symbols {syms}")
    else:
        cp = syms[:max_corner]
    omitted = [s for s in syms if s not in cp]
    idx = [syms.index(s) for s in cp]
    files["corner_png"] = plot_corner(samples[:, idx], cp, out_prefix + "_corner.png")
    files["correlation_png"] = plot_correlation(st["correlation"], syms, out_prefix + "_correlation.png")
    # tables
    rows = []
    for p in st["parameters"]:
        rows.append({"symbol": p["symbol"], "tdb_parameters": "; ".join(p["tdb_parameters"]),
                     "value_in_tdb": p["value_in_tdb"], "mean": p["mean"], "median": p["median"], "std": p["std"],
                     "ci95_low": p["ci95"][0], "ci95_high": p["ci95"][1], "map": p["map"], "tau": p["tau"],
                     "tau_reliable": p["tau_reliable"], "ess": p["ess"], "split_rhat": p["rhat"]})
    files["summary_csv"] = out_prefix + "_summary.csv"
    with open(files["summary_csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"trace": str(trace), "lnprob": str(lnprob), "tdb": str(tdb),
               "allocated_iterations": ch["allocated_iterations"], "completed_iterations": ch["completed_iterations"],
               "zero_filled_iterations_removed": ch["allocated_iterations"] - ch["completed_iterations"],
               "truncation": ch["truncation"], "burn_in_info": binfo, "symbols": syms,
               "corner_parameters": cp, "corner_omitted": omitted,
               **{k: val for k, val in st.items() if k != "correlation"},
               "correlation_matrix": np.where(np.isfinite(st["correlation"]), st["correlation"], None).tolist(),
               "files": files}
    files["summary_json"] = out_prefix + "_summary.json"
    Path(files["summary_json"]).write_text(json.dumps(summary, indent=1, default=str))
    summary["text"] = _analysis_text(summary)
    return summary


def _analysis_text(s: Dict) -> str:
    acc = s["acceptance_fraction"]
    b = s["burn_in_info"]
    L = [f"chains {s['n_chains']}, parameters {s['n_parameters']}, iterations completed "
         f"{s['completed_iterations']}/{s['allocated_iterations']}"
         + (f" ({s['zero_filled_iterations_removed']} zero-filled iterations removed: unfinished run)"
            if s["zero_filled_iterations_removed"] else ""),
         f"burn-in {s['burn_in']} ({b['method']}"
         + (f"; stable lnprob ~ {b['reference_lnprob']:.4g} +- {b['tolerance']:.3g}" if "reference_lnprob" in b else "")
         + f"), thin {s['thin']} -> {s['n_samples']} posterior samples",
         f"acceptance fraction mean {acc['mean']:.3f} (min {acc['min']:.3f}, max {acc['max']:.3f}; "
         f"{acc['stuck_chains']} stuck chains)",
         f"MAP lnprob {s['map']['lnprob']:.5g} (chain {s['map']['chain']}, iteration {s['map']['iteration']}); "
         f"final median lnprob {s['lnprob_final_median']:.5g}", "",
         "symbol | TDB parameter [coefficient of] | TDB value | mean | median | std | 95% CI | tau | ESS | R-hat"]
    for p in s["parameters"]:
        tau = f"{p['tau']:.3g}{'' if p['tau_reliable'] else '*'}" if np.isfinite(p["tau"]) else "nan"
        L.append(f"{p['symbol']} | {'; '.join(p['tdb_parameters']) or '-'} | {_fmt(p['value_in_tdb'])} | "
                 f"{_fmt(p['mean'])} | {_fmt(p['median'])} | {p['std']:.4g} | [{_fmt(p['ci95'][0])}, "
                 f"{_fmt(p['ci95'][1])}] | {tau} | {p['ess']:.0f} | {p['rhat']:.3f}")
    L.append("(* = unreliable: post-burn-in chain shorter than 50 tau; R-hat = split Gelman-Rubin over walkers, "
             "indicative only for an ensemble sampler)")
    if s["strong_correlations"]:
        L.append("strong correlations |r|>=0.8: " + ", ".join(
            f"{c['pair'][0]}-{c['pair'][1]} {c['r']:+.2f}" for c in s["strong_correlations"][:12]))
    if s["corner_omitted"]:
        L.append(f"corner plot shows {len(s['corner_parameters'])} parameters; omitted {s['corner_omitted']} "
                 "(choose with corner_parameters)")
    if s["warnings"]:
        L.append("WARNINGS:\n  - " + "\n  - ".join(s["warnings"]))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# parameter sets -> TDB
# ---------------------------------------------------------------------------
def parameter_set(tr: np.ndarray, lp: np.ndarray, burn: int, statistic: str) -> np.ndarray:
    st = statistic.lower()
    if st == "map":
        flat = lp.reshape(-1)
        i = int(np.nanargmax(flat))
        c, it = divmod(i, lp.shape[1])
        return tr[c, it, :]
    samples = posterior_samples(tr, burn, 1)
    if st == "mean":
        return samples.mean(axis=0)
    if st == "median":
        return np.median(samples, axis=0)
    raise ValueError("statistic must be mean, median or map")


def mcmc_to_tdb(tdb: str, trace: str, lnprob: str, output: str, *, statistic: str = "median",
                burn_in: Union[int, str, None] = "auto", symbols: Optional[Sequence[str]] = None) -> Dict:
    ch = load_chains(trace, lnprob)
    tr, lp = ch["trace"], ch["lnprob"]
    db = _load_db(tdb)
    syms = fitted_symbols(db, tr.shape[2], symbols)
    burn, binfo = _burn(burn_in, lp)
    vals = parameter_set(tr, lp, burn, statistic)
    usage = symbol_usage(db, syms)
    old = {s: usage[s]["value_in_tdb"] for s in syms}
    for s, val in zip(syms, vals):
        db.symbols[s] = float(val)        # same as ESPEI's optimizer: dbf.symbols.update(parameters)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db.to_file(output, if_exists="overwrite")
    back = _load_db(output)
    check = max((abs(_unpack(back.symbols[s]) - float(v)) / max(abs(float(v)), 1e-300) for s, v in zip(syms, vals)),
                default=0.0)
    rows = [{"symbol": s, "tdb_parameters": _describe(usage, s), "old": old[s], "new": float(v),
             "change": (float(v) - old[s]) if old[s] is not None else None} for s, v in zip(syms, vals)]
    return {"output": output, "statistic": statistic.lower(), "burn_in": burn, "burn_in_info": binfo,
            "parameters": rows, "reload_max_relative_error": check,
            "completed_iterations": ch["completed_iterations"]}


# ---------------------------------------------------------------------------
# uncertainty propagation
# ---------------------------------------------------------------------------
def _sample_db(base_db, syms: Sequence[str], vals: Sequence[float]):
    db = copy.deepcopy(base_db)
    db.symbols.update({s: float(v) for s, v in zip(syms, vals)})
    return db


def _draw(tr, lp, burn, n, seed) -> Tuple[np.ndarray, np.ndarray]:
    samples = posterior_samples(tr, burn, 1)
    rng = np.random.default_rng(seed)
    uniq, first = np.unique(samples, axis=0, return_index=True)
    pool = samples[np.sort(first)] if len(uniq) >= n else samples
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return pool[idx], samples


def _map_binary_sample(db, els, phases, tlo, thi, dT, dx, time_limit) -> Dict:
    from pycalphad import variables as v
    from pycalphad.mapping import BinaryStrategy
    from . import mapping_tools as mt
    comps = sorted(els) + ["VA"]
    conds = {v.P: 101325.0, v.N: 1, v.T: (tlo, thi, dT), v.X(els[1]): (0.0, 1.0, dx)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strategy = BinaryStrategy(db, comps, phases, conds)
    run = mt._run_strategy(strategy, time_limit)
    lab = mt._Labeler(db, strategy.models, True)
    s_els = [e for e in strategy.elements if e != "VA"]
    with tempfile.TemporaryDirectory() as td:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mt._tieline_data(strategy, db, comps, phases, els, "binary", s_els, lab, mt._Units("K"),
                                   Path(td), "s", {})
    b = els[1]
    segs = []
    for ln in res["lines"]:
        npx = len(ln["phases"])
        for k in range(npx):
            xs, Ts, name = [], [], ln["phases"][k]
            for p in ln["points"]:
                if k < len(p["phases"]):
                    xs.append(p["phases"][k][1][b])
                    Ts.append(p["T"])
            if len(xs) > 1:
                segs.append({"phase": name, "region": ln["region"], "x": xs, "T": Ts})
    invs = [{"T_K": i["T_K"], "phases": sorted(i["phases"]), "type": i.get("type", ""),
             "reaction": i.get("reaction_on_cooling", ""),
             "X": {n: i["compositions"][n][b] for n in i["phases"]}} for i in res["invariants"]]
    return {"finished": run["finished"], "seconds": run["seconds"], "segments": segs, "invariants": invs,
            "phases_found": sorted(res["phases_found"])}


def _stats(a: Sequence[float]) -> Dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    q = np.percentile(a, [2.5, 50, 97.5])
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(q[1]),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "ci95": [float(q[0]), float(q[2])],
            "min": float(a.min()), "max": float(a.max())}


def propagate_binary(tdb: str, trace: str, lnprob: str, elements: Sequence[str], out_prefix: str, *,
                     tmin: float, tmax: float, temperature_unit: str = "K", phases=None,
                     burn_in: Union[int, str, None] = "auto", n_samples: int = 30, seed: int = 0,
                     composition_step: float = 0.05, temperature_step: float = 20.0,
                     time_limit_s: float = 600.0, per_sample_time_limit_s: float = 60.0,
                     symbols: Optional[Sequence[str]] = None) -> Dict:
    from . import mapping_tools as mt
    t0 = time.time()
    ch = load_chains(trace, lnprob)
    tr, lp = ch["trace"], ch["lnprob"]
    base = _load_db(tdb)
    syms = fitted_symbols(base, tr.shape[2], symbols)
    els = mt._check_elements(base, elements)
    if len(els) != 2:
        raise ValueError("binary_diagram needs exactly 2 elements")
    ph = mt._select_phases(base, els, phases, None)
    burn, binfo = _burn(burn_in, lp)
    n_req = int(n_samples)
    if n_req < 1:
        raise ValueError("n_samples must be >= 1")
    n_use = min(n_req, MAX_SAMPLES)
    draws, post = _draw(tr, lp, burn, n_use, seed)
    unit = "C" if str(temperature_unit).upper().startswith("C") else "K"
    tlo, thi = sorted((mt._toK(tmin, unit), mt._toK(tmax, unit)))
    med_vals = np.median(post, axis=0)
    ref = _map_binary_sample(_sample_db(base, syms, med_vals), els, ph, tlo, thi, temperature_step,
                             composition_step, per_sample_time_limit_s)
    results, stopped = [], None
    for i, vals in enumerate(draws):
        if time.time() - t0 > time_limit_s:
            stopped = f"time limit {time_limit_s:g} s reached after {i} of {len(draws)} samples"
            break
        try:
            r = _map_binary_sample(_sample_db(base, syms, vals), els, ph, tlo, thi, temperature_step,
                                   composition_step, min(per_sample_time_limit_s, max(5.0, time_limit_s -
                                                                                      (time.time() - t0))))
            r["ok"] = True
        except Exception as e:  # noqa: BLE001 - a failed sample must not stop the others
            r = {"ok": False, "error": f"{type(e).__name__}: {e}", "segments": [], "invariants": []}
        r["index"] = i
        results.append(r)
    ok = [r for r in results if r["ok"]]
    # invariant distributions (grouped by phase set)
    groups: "OrderedDict[str, Dict]" = OrderedDict()
    for inv in ref["invariants"]:
        groups.setdefault(" + ".join(inv["phases"]), {"reference": inv, "T": [], "X": {}, "types": []})
    for r in ok:
        seen = set()
        for inv in r["invariants"]:
            key = " + ".join(inv["phases"])
            if key in seen:
                continue
            seen.add(key)
            g = groups.setdefault(key, {"reference": None, "T": [], "X": {}, "types": []})
            g["T"].append(inv["T_K"])
            g["types"].append(inv["type"])
            for n, x in inv["X"].items():
                g["X"].setdefault(n, []).append(x)
    inv_summary = []
    for key, g in groups.items():
        st = _stats(g["T"])
        typ = Counter(g["types"]).most_common(1)[0][0] if g["types"] else (g["reference"] or {}).get("type", "")
        inv_summary.append({
            "phases": key, "type": typ, "found_in_samples": len(g["T"]), "n_samples": len(ok),
            "T_K": st, "T_C": {k: ([x - KELVIN for x in val] if isinstance(val, list) else
                                   (val - KELVIN if k in ("mean", "median", "min", "max") else val))
                               for k, val in st.items()},
            "T_K_median_parameters": (g["reference"] or {}).get("T_K"),
            f"X({els[1]})": {n: _stats(xs) for n, xs in g["X"].items()}})
    # ------------------------------------------------ plot
    U = mt._Units(unit)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    alpha = float(np.clip(3.0 / max(len(ok), 1), 0.06, 0.4))
    found = set(ref["phases_found"])
    for r in ok:
        found |= set(r["phases_found"])
        for sg in r["segments"]:
            ax.plot(sg["x"], [U.T(t) for t in sg["T"]], color=mt.phase_color(sg["phase"]), lw=0.7, alpha=alpha)
    for sg in ref["segments"]:
        ax.plot(sg["x"], [U.T(t) for t in sg["T"]], color=mt.phase_color(sg["phase"]), lw=1.9)
    for inv in ref["invariants"]:
        xs = list(inv["X"].values())
        ax.plot([min(xs), max(xs)], [U.T(inv["T_K"])] * 2, color=mt.INVARIANT_COLOR, lw=1.5)
    for g in inv_summary:
        st = g["T_K"]
        if st.get("n", 0) >= 2:
            ax.axhspan(U.T(st["ci95"][0]), U.T(st["ci95"][1]), color=mt.INVARIANT_COLOR, alpha=0.08, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(U.T(tlo), U.T(thi))
    ax.set_xlabel(U.comp_label(els[1]))
    ax.set_ylabel(U.T_label())
    ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=mt.phase_color(p), label=p) for p in sorted(found)]
    handles += [Line2D([], [], color="0.3", lw=1.9, label="posterior-median parameters"),
                Line2D([], [], color="0.3", lw=0.7, alpha=0.5, label=f"posterior samples ({len(ok)})")]
    if inv_summary:
        handles.append(Patch(color=mt.INVARIANT_COLOR, alpha=0.2, label="invariant T, 95 % interval"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    system = "-".join(mt._cap(e) for e in els)
    ax.set_title(f"{system}: posterior uncertainty of the phase diagram" + ("  [PARTIAL]" if stopped else ""),
                 fontsize=11)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    files = {"png": out_prefix + ".png"}
    fig.savefig(files["png"], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    # invariant histograms
    hist_groups = [g for g in inv_summary if g["T_K"].get("n", 0) >= 2][:6]
    if hist_groups:
        rows, cols = _grid(len(hist_groups), 3)
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
        for idx in range(rows * cols):
            a = axes[idx // cols][idx % cols]
            if idx >= len(hist_groups):
                a.set_visible(False)
                continue
            g = hist_groups[idx]
            src = groups[g["phases"]]
            Ts = [U.T(t) for t in src["T"]]
            a.hist(Ts, bins=min(20, max(5, len(Ts) // 2)), color="#fcbba1", edgecolor="#cb181d")
            st = g["T_K"]
            a.axvline(U.T(st["median"]), color="#cb181d", lw=1.2, label="median")
            a.axvline(U.T(st["ci95"][0]), color="#cb181d", lw=0.8, ls="--", label="95 % interval")
            a.axvline(U.T(st["ci95"][1]), color="#cb181d", lw=0.8, ls="--")
            if g["T_K_median_parameters"] is not None:
                a.axvline(U.T(g["T_K_median_parameters"]), color="k", lw=1.0, ls=":",
                          label="median parameters")
            a.set_title(f"{g['type'] or 'invariant'}: {g['phases']}\n"
                        f"{g['found_in_samples']}/{g['n_samples']} samples", fontsize=8)
            a.set_xlabel(U.T_label(), fontsize=8)
            a.set_ylabel("Samples", fontsize=8)
            a.yaxis.set_major_locator(MaxNLocator(integer=True))
            a.tick_params(labelsize=7)
            a.legend(fontsize=6)
        fig.tight_layout()
        files["invariants_png"] = out_prefix + "_invariants.png"
        fig.savefig(files["invariants_png"], dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    # data files
    files["boundaries_csv"] = out_prefix + "_boundaries.csv"
    with open(files["boundaries_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "segment", "phase", "region", f"X({els[1]})", "T_K", "T_C"])
        for tag, r in [("median_parameters", ref)] + [(str(r["index"]), r) for r in ok]:
            for si, sg in enumerate(r["segments"]):
                for x, t in zip(sg["x"], sg["T"]):
                    w.writerow([tag, si, sg["phase"], sg["region"], f"{x:.6g}", f"{t:.4f}", f"{t - KELVIN:.4f}"])
    files["samples_csv"] = out_prefix + "_parameter_samples.csv"
    with open(files["samples_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample"] + list(syms))
        w.writerow(["median_parameters"] + [f"{x:.10g}" for x in med_vals])
        for i, vals in enumerate(draws[:len(results)]):
            w.writerow([i] + [f"{x:.10g}" for x in vals])
    secs = time.time() - t0
    summary = {"kind": "binary_diagram", "system": system, "tdb": str(tdb), "phases": ph,
               "burn_in": burn, "burn_in_info": binfo, "n_samples_requested": n_req, "n_samples_drawn": len(draws),
               "n_samples_computed": len(results), "n_samples_ok": len(ok),
               "n_samples_unfinished": sum(1 for r in ok if not r["finished"]),
               "failed": [{"sample": r["index"], "error": r["error"]} for r in results if not r["ok"]][:10],
               "stopped": stopped, "resolution": {"composition_step": composition_step,
                                                  "temperature_step_K": temperature_step},
               "invariants": inv_summary, "seconds": round(secs, 1),
               "seconds_per_sample": round(float(np.mean([r["seconds"] for r in ok])), 2) if ok else None,
               "files": files}
    if n_req > MAX_SAMPLES:
        summary["note"] = f"n_samples capped at {MAX_SAMPLES}"
    Path(out_prefix + "_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    files["summary_json"] = out_prefix + "_summary.json"
    return summary


def _crossing(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """First x (in axis order) at which y crosses thr (linear interpolation)."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    s = np.sign(y - thr)
    for i in range(len(x) - 1):
        if s[i] == 0:
            return float(x[i])
        if s[i] * s[i + 1] < 0:
            return float(x[i] + (thr - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


def propagate_property(tdb: str, trace: str, lnprob: str, elements: Sequence[str], out_prefix: str, *,
                       quantity: str, conditions: Dict, temperature_unit: str = "K", phases=None,
                       burn_in: Union[int, str, None] = "auto", n_samples: int = 30, seed: int = 0,
                       time_limit_s: float = 600.0, threshold: Optional[float] = None,
                       symbols: Optional[Sequence[str]] = None) -> Dict:
    from . import mapping_tools as mt
    from . import props
    t0 = time.time()
    ch = load_chains(trace, lnprob)
    tr, lp = ch["trace"], ch["lnprob"]
    base = _load_db(tdb)
    syms = fitted_symbols(base, tr.shape[2], symbols)
    els = mt._check_elements(base, elements)
    ph = mt._select_phases(base, els, phases, None)
    _, _, varying, values = props.parse_conditions(conditions, els, temperature_unit)
    if len(varying) != 1:
        raise ValueError("property propagation needs exactly one varying condition, e.g. "
                         "{\"T\": [1200, 2000, 10], \"X(NI)\": 0.5}")
    xlab = varying[0]
    burn, binfo = _burn(burn_in, lp)
    n_req = int(n_samples)
    if n_req < 1:
        raise ValueError("n_samples must be >= 1")
    draws, post = _draw(tr, lp, burn, min(n_req, MAX_SAMPLES), seed)
    med_vals = np.median(post, axis=0)

    def run(vals):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = props.evaluate_quantities(_sample_db(base, syms, vals), els, ph, conditions, [quantity],
                                            unit=temperature_unit)
        if xlab not in res["axis_labels"]:
            raise ValueError("could not identify the varying axis")
        return {lab: np.asarray(arr, float).reshape(-1) for lab, arr in res["columns"].items()}, res["units"]

    ref, units = run(med_vals)
    curves: Dict[str, List[np.ndarray]] = {lab: [] for lab in ref}
    stopped, failed = None, []
    for i, vals in enumerate(draws):
        if time.time() - t0 > time_limit_s:
            stopped = f"time limit {time_limit_s:g} s reached after {i} of {len(draws)} samples"
            break
        try:
            cols, _ = run(vals)
        except Exception as e:  # noqa: BLE001
            failed.append({"sample": i, "error": f"{type(e).__name__}: {e}"})
            continue
        for lab in curves:
            curves[lab].append(cols.get(lab, np.full_like(ref[lab], np.nan)))
    x = props._cond_display(xlab, np.asarray(values[xlab], float), temperature_unit)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    band_rows, stats = {}, {}
    n_ok = 0
    for ci, (lab, arrs) in enumerate(curves.items()):
        col = colors[ci % len(colors)]
        A = np.array(arrs) if arrs else np.empty((0, len(x)))
        n_ok = max(n_ok, len(A))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(A, axis=0) if len(A) else np.full(len(x), np.nan)
            lo = np.nanpercentile(A, 2.5, axis=0) if len(A) else med
            hi = np.nanpercentile(A, 97.5, axis=0) if len(A) else med
        for a in A:
            ax.plot(x, a, color=col, lw=0.5, alpha=float(np.clip(2.0 / max(len(A), 1), 0.04, 0.3)))
        ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0, label=f"{lab}: 95 % band ({len(A)} samples)")
        ax.plot(x, med, color=col, lw=1.8, label=f"{lab}: median over samples")
        ax.plot(x, ref[lab], color=col, lw=1.0, ls="--", label=f"{lab}: posterior-median parameters")
        band_rows[lab] = (med, lo, hi, A)
        if threshold is not None:
            cr = [_crossing(x, a, float(threshold)) for a in A]
            stats[lab] = {"threshold": float(threshold), "crossing_" + xlab: _stats(cr),
                          "crossing_median_parameters": _crossing(x, ref[lab], float(threshold)),
                          "samples_without_crossing": int(np.sum(~np.isfinite(cr)))}
    ax.set_xlabel(props._cond_axis_label(xlab, temperature_unit))
    unit_s = next(iter(units.values()), "")
    ax.set_ylabel(f"{', '.join(curves)}" + (f" ({unit_s})" if unit_s else ""))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fixed = ", ".join(f"{k}={props._cond_display(k, v_, temperature_unit)[0]:g}" for k, v_ in values.items()
                      if k != xlab and k not in ("N",) and not (k == "P" and v_[0] == 101325.0))
    ax.set_title(f"{'-'.join(mt._cap(e) for e in els)}: posterior uncertainty of {quantity}"
                 + (f" ({fixed})" if fixed else "") + ("  [PARTIAL]" if stopped else ""), fontsize=10)
    files = {"png": out_prefix + ".png"}
    fig.tight_layout()
    fig.savefig(files["png"], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    files["band_csv"] = out_prefix + "_band.csv"
    with open(files["band_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        head = [xlab + (" (C)" if xlab == "T" and props._is_celsius(temperature_unit) else "")]
        for lab in band_rows:
            head += [f"{lab} median", f"{lab} p2.5", f"{lab} p97.5", f"{lab} median_parameters"]
        w.writerow(head)
        for i in range(len(x)):
            row = [f"{x[i]:.6g}"]
            for lab, (med, lo, hi, _) in band_rows.items():
                row += [f"{med[i]:.6g}", f"{lo[i]:.6g}", f"{hi[i]:.6g}", f"{ref[lab][i]:.6g}"]
            w.writerow(row)
    files["samples_csv"] = out_prefix + "_samples.csv"
    with open(files["samples_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "sample"] + [f"{xv:.6g}" for xv in x])
        for lab, (_, _, _, A) in band_rows.items():
            for i, a in enumerate(A):
                w.writerow([lab, i] + [f"{val:.6g}" for val in a])
    widths = {lab: float(np.nanmax(hi - lo)) if np.isfinite(hi - lo).any() else float("nan")
              for lab, (_, lo, hi, _) in band_rows.items()}
    secs = time.time() - t0
    summary = {"kind": "property", "quantity": quantity, "axis": xlab, "units": units, "tdb": str(tdb),
               "phases": ph, "burn_in": burn, "burn_in_info": binfo, "n_samples_requested": n_req,
               "n_samples_drawn": len(draws), "n_samples_ok": n_ok, "failed": failed[:10], "stopped": stopped,
               "max_95_band_width": widths, "threshold_crossings": stats, "seconds": round(secs, 1),
               "files": files}
    if n_req > MAX_SAMPLES:
        summary["note"] = f"n_samples capped at {MAX_SAMPLES}"
    Path(out_prefix + "_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    files["summary_json"] = out_prefix + "_summary.json"
    return summary


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
def register(tb) -> None:  # noqa: C901 - flat list of tool definitions
    S = {"type": "string"}
    N = {"type": "number"}
    I = {"type": "integer"}
    STRS = {"type": "array", "items": {"type": "string"}}
    BURN = {"type": "string", "description": "number of iterations to discard (e.g. \"200\"), or \"auto\" (default: "
                                             "detected from lnprob stabilisation)"}
    RUN = {"trace": {**S, "description": "ESPEI trace.npy (or the run directory containing trace.npy, lnprob.npy "
                                         "and mcmc.tdb/dft.tdb)"},
           "lnprob": {**S, "description": "ESPEI lnprob.npy (default: next to the trace)"},
           "tdb": {**S, "description": "TDB with the fitted VV#### symbols (MCMC input dft.tdb or output mcmc.tdb; "
                                       "default: next to the trace)"},
           "symbols": {**STRS, "description": "only if the ESPEI YAML had mcmc.symbols: that list"}}

    def files_(trace, lnprob, tdb):
        tp = tb.p(trace)
        t, lpp, d = resolve_run_files(tp, tb.p(lnprob) if lnprob else None,
                                      tb._find_tdb(tdb) if tdb else None)
        return t, lpp, d

    def prefix(output_prefix, default):
        p = tb.outp(output_prefix, default)
        return str(Path(p).with_suffix("")) if Path(p).suffix.lower() in (".png", ".json", ".csv") else p

    def stem(t):
        pt = Path(t)
        return pt.parent.name or pt.stem

    @tb.add("analyze_mcmc",
            "Analyse an ESPEI MCMC run (trace.npy + lnprob.npy + the TDB with the VV#### symbols): removes "
            "zero-filled iterations of unfinished runs, detects burn-in (lnprob stabilisation) unless given, "
            "reports per-parameter mean/median/std/95% credible interval/MAP, acceptance fraction, integrated "
            "autocorrelation time + effective sample size, split R-hat, strong correlations, and which TDB "
            "parameter (and coefficient: const, T, T*LN(T)) each symbol belongs to. Plots: lnprob vs iteration, "
            "parameter traces, corner plot, correlation matrix; CSV/JSON tables.",
            {**RUN, "burn_in": BURN, "thin": {**I, "description": "keep every n-th post-burn-in iteration (default 1)"},
             "corner_parameters": {**STRS, "description": "symbols for the corner plot (default: first 12)"},
             "output_prefix": {**S, "description": "file stem for the outputs"}},
            ["trace"])
    def _analyze(trace, lnprob=None, tdb=None, symbols=None, burn_in="auto", thin=1, corner_parameters=None,
                 output_prefix=None):
        t, lpp, d = files_(trace, lnprob, tdb)
        out = prefix(output_prefix, f"mcmc_analysis/{stem(t)}")
        r = analyze_mcmc(t, lpp, d, out, burn_in=burn_in, thin=thin or 1, symbols=symbols,
                         corner_parameters=corner_parameters)
        fl = "\n".join(f"  {k}: {tb.rel(v)}" for k, v in r["files"].items())
        imgs = [r["files"][k] for k in ("lnprob_png", "traces_png", "corner_png", "correlation_png")]
        return f"TDB: {tb.rel(d)}\n{r['text']}\n\nfiles:\n{fl}", imgs

    @tb.add("mcmc_to_tdb",
            "Write a TDB with a parameter set from an ESPEI MCMC run substituted for the fitted VV#### symbols: "
            "posterior median (default), mean (after burn-in) or map (maximum lnprob over the whole run, as ESPEI's "
            "own mcmc.tdb). Reports old/new values per symbol with the TDB parameter it belongs to.",
            {**RUN, "statistic": {"type": "string", "enum": ["median", "mean", "map"]}, "burn_in": BURN,
             "output": {**S, "description": "output TDB file name"}},
            ["trace"])
    def _to_tdb(trace, lnprob=None, tdb=None, symbols=None, statistic="median", burn_in="auto", output=None):
        t, lpp, d = files_(trace, lnprob, tdb)
        out = tb.outp(output if (not output or output.lower().endswith(".tdb")) else output + ".tdb",
                      f"{stem(t)}_{statistic}.tdb")
        r = mcmc_to_tdb(d, t, lpp, out, statistic=statistic, burn_in=burn_in, symbols=symbols)
        L = [f"wrote {tb.rel(out)} ({r['statistic']}, burn-in {r['burn_in']} of {r['completed_iterations']} "
             f"iterations; reload check max rel. error {r['reload_max_relative_error']:.1e})",
             "symbol | TDB parameter [coefficient of] | old | new | change"]
        for row in r["parameters"]:
            L.append(f"{row['symbol']} | {row['tdb_parameters']} | {_fmt(row['old'])} | {_fmt(row['new'])} | "
                     f"{_fmt(row['change'])}")
        return "\n".join(L), []

    @tb.add("propagate_uncertainty",
            "Propagate ESPEI MCMC parameter uncertainty: draw n_samples posterior parameter sets (after burn-in) and "
            "recompute (kind=binary_diagram) the binary phase diagram for each (coarse mapping; all sample "
            "boundaries semi-transparent + the posterior-median-parameter diagram bold; distribution of every "
            "invariant temperature, e.g. the eutectic T, with 95% interval) or (kind=property) one equilibrium "
            "quantity along one varying condition (median + 95% band; optional threshold -> distribution of the "
            "crossing point, e.g. NP(LIQUID) with threshold 0.99 ~ liquidus). A binary diagram costs ~1-10 s per "
            "sample; n_samples is capped at 200 and the run stops at time_limit_s (partial result).",
            {**RUN, "kind": {"type": "string", "enum": ["binary_diagram", "property"]}, "elements": STRS,
             "phases": STRS, "burn_in": BURN,
             "n_samples": {**I, "description": "posterior samples (default 30, max 200)"},
             "seed": I, "tmin": N, "tmax": N,
             "temperature_unit": {"type": "string", "enum": ["K", "C"]},
             "composition_step": {**N, "description": "binary mapping step (default 0.05)"},
             "temperature_step": {**N, "description": "binary mapping step in K (default 20)"},
             "quantity": {**S, "description": "property kind: e.g. NP(LIQUID), X(FCC_A1,CR), HM, MU(NI), CPM"},
             "conditions": {"type": "object", "description": "property kind: one varying condition, e.g. "
                                                             "{\"T\": [1400, 2000, 10], \"X(NI)\": 0.5}"},
             "threshold": {**N, "description": "property kind: report where the quantity crosses this value"},
             "time_limit_s": {**N, "description": "total time budget (default 600 s)"},
             "output_prefix": S},
            ["trace", "kind", "elements"])
    def _propagate(trace, kind, elements, lnprob=None, tdb=None, symbols=None, phases=None, burn_in="auto",
                   n_samples=30, seed=0, tmin=None, tmax=None, temperature_unit="K", composition_step=0.05,
                   temperature_step=20.0, quantity=None, conditions=None, threshold=None, time_limit_s=600.0,
                   output_prefix=None):
        t, lpp, d = files_(trace, lnprob, tdb)
        els = "-".join(e.upper() for e in elements)
        if kind == "binary_diagram":
            if tmin is None or tmax is None:
                raise ValueError("binary_diagram needs tmin and tmax")
            out = prefix(output_prefix, f"uncertainty/{stem(t)}_{els}_diagram")
            r = propagate_binary(d, t, lpp, elements, out, tmin=tmin, tmax=tmax, temperature_unit=temperature_unit,
                                 phases=phases, burn_in=burn_in, n_samples=n_samples, seed=seed or 0,
                                 composition_step=composition_step, temperature_step=temperature_step,
                                 time_limit_s=time_limit_s, symbols=symbols)
            L = [f"{r['system']} diagram for {r['n_samples_ok']}/{r['n_samples_drawn']} posterior samples "
                 f"(+ posterior-median parameters) in {r['seconds']} s ({r['seconds_per_sample']} s/sample; "
                 f"burn-in {r['burn_in']})"]
            if r.get("stopped"):
                L.append("STOPPED: " + r["stopped"])
            if r["n_samples_unfinished"]:
                L.append(f"{r['n_samples_unfinished']} sample map(s) hit the per-sample time limit (partial)")
            for f in r["failed"]:
                L.append(f"sample {f['sample']} failed: {f['error']}")
            L.append("invariants (T in K; median [95% interval], found in n samples):")
            for g in r["invariants"]:
                st = g["T_K"]
                ref = g["T_K_median_parameters"]
                if st.get("n"):
                    L.append(f"  {g['type'] or 'invariant'} {g['phases']}: {st['median']:.1f} K "
                             f"[{st['ci95'][0]:.1f}, {st['ci95'][1]:.1f}] (std {st['std']:.1f}; "
                             f"{st['median'] - KELVIN:.1f} °C), {g['found_in_samples']}/{g['n_samples']} samples; "
                             f"median-parameter diagram: {'absent' if ref is None else f'{ref:.1f} K'}")
                else:
                    L.append(f"  {g['type']} {g['phases']}: only in the median-parameter diagram ({ref:.1f} K)")
            if "note" in r:
                L.append(r["note"])
        elif kind == "property":
            if not quantity or not conditions:
                raise ValueError("property kind needs `quantity` and `conditions`")
            q = re.sub(r"[^A-Za-z0-9]+", "_", quantity).strip("_")
            out = prefix(output_prefix, f"uncertainty/{stem(t)}_{els}_{q}")
            r = propagate_property(d, t, lpp, elements, out, quantity=quantity, conditions=conditions,
                                   temperature_unit=temperature_unit, phases=phases, burn_in=burn_in,
                                   n_samples=n_samples, seed=seed or 0, time_limit_s=time_limit_s,
                                   threshold=threshold, symbols=symbols)
            L = [f"{quantity} vs {r['axis']} for {r['n_samples_ok']}/{r['n_samples_drawn']} posterior samples in "
                 f"{r['seconds']} s (burn-in {r['burn_in']})",
                 "max width of the 95% band: " + ", ".join(f"{k} {v:.4g}" for k, v in r["max_95_band_width"].items())]
            if r.get("stopped"):
                L.append("STOPPED: " + r["stopped"])
            for f in r["failed"]:
                L.append(f"sample {f['sample']} failed: {f['error']}")
            for lab, st in r["threshold_crossings"].items():
                c = st["crossing_" + r["axis"]]
                if c.get("n"):
                    L.append(f"{lab} crosses {st['threshold']:g} at {r['axis']} = {c['median']:.5g} "
                             f"[{c['ci95'][0]:.5g}, {c['ci95'][1]:.5g}] (std {c['std']:.3g}; median parameters: "
                             f"{st['crossing_median_parameters']:.5g}; no crossing in "
                             f"{st['samples_without_crossing']} samples)")
                else:
                    L.append(f"{lab} never crosses {st['threshold']:g}")
            if "note" in r:
                L.append(r["note"])
        else:
            raise ValueError("kind must be binary_diagram or property")
        fl = "\n".join(f"  {k}: {tb.rel(v)}" for k, v in r["files"].items())
        imgs = [r["files"]["png"]] + ([r["files"]["invariants_png"]] if "invariants_png" in r["files"] else [])
        return "\n".join(L) + f"\nfiles:\n{fl}", imgs
