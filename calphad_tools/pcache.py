"""Process-wide cache of pycalphad objects, shared by every calculation tool and ToolBox in this process.

What is cached (LRU, capped):

* ``Database`` objects, keyed by file *content* (sha256) and file type; the hash is only recomputed when the
  file's (path, mtime_ns, size) changes, so an edited TDB is re-read and gets fresh entries, and an edited copy
  of a TDB gets its own entries.
* ``Model`` instances per (database, components, phase, model class, parameter symbols).
* Compiled callables of ``PhaseRecordFactory`` (the symengine/LLVM code generation that dominates repeated
  calls), shared between phase subsets of the same (database, components, state variables, model class,
  parameter symbols) "family".  Each call still gets a ``PhaseRecordFactory`` whose ``models`` are exactly the
  requested phases, so wildcard properties (``NP(*)``) and phase lists are unchanged.

How it is used: calculation code runs inside ``with pcache.use():``.  While active, pycalphad's
``instantiate_models`` / ``PhaseRecordFactory`` references in ``Workspace``, ``calculate``, the mapping
strategies and ``scheil`` are routed through this cache.  Only databases loaded through :func:`get_db` are
cached; any other database, custom Model instances or parameter *values* fall back to plain pycalphad (compiled
functions depend on parameter symbols only, so a factory with parameter values is always a fresh object).

Safety: a cheap fingerprint of each cached Database (phases, parameter/symbol counts and expression hashes) is
checked on every use; if code mutated a shared Database, its entries are dropped and it is no longer cached.
Cached Models / factories are never mutated by pycalphad's equilibrium, calculate, mapping or scheil code.

Generated standalone code (show_code) never uses this module.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_LOCK = threading.RLock()

LIMITS = {"databases": 16, "models": 1500, "families": 32, "factories": 128}


class _LRU:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.d: "OrderedDict[Any, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with _LOCK:
            if key in self.d:
                self.d.move_to_end(key)
                self.hits += 1
                return self.d[key]
            self.misses += 1
            return None

    def put(self, key, val):
        with _LOCK:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)
        return val

    def drop(self, pred):
        with _LOCK:
            for k in [k for k in self.d if pred(k)]:
                del self.d[k]

    def clear(self):
        with _LOCK:
            self.d.clear()
            self.hits = self.misses = 0


_STAT: Dict[str, Tuple[int, int, str, int]] = {}     # resolved path -> (mtime_ns, size, key, hashed_at_ns)
_DBS = _LRU(LIMITS["databases"])                       # content key -> (Database, fingerprint, path)
_OWNED: Dict[int, str] = {}                            # id(Database) -> content key (only live cache entries)
_MODELS = _LRU(LIMITS["models"])                       # (dbkey, comps, phase, cls, psyms) -> Model
_FAMILIES = _LRU(LIMITS["families"])                   # (dbkey, comps, cls, psyms, statevars) -> compiled store
_FACTORIES = _LRU(LIMITS["factories"])                 # family key + phases -> _SharedPRF (no parameter values)
_TIMES = {"parse_s": 0.0, "models_s": 0.0}
_ENABLED = os.environ.get("CALPHAD_TOOLS_NO_CACHE", "") == ""


# ---------------------------------------------------------------------------------------------------------
# databases
# ---------------------------------------------------------------------------------------------------------
def file_key(path: str) -> str:
    """Content key of a database file ("<suffix>:<sha256>").

    Re-hashed when the file's mtime or size changes, and always while the last hash was taken less than
    2 s after the file's mtime (an edit within the same timestamp tick would otherwise go unnoticed)."""
    p = str(Path(path).resolve())
    st = os.stat(p)
    with _LOCK:
        hit = _STAT.get(p)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size and hit[3] - st.st_mtime_ns > 2e9:
            return hit[2]
    hashed_at = time.time_ns()
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    key = f"{Path(p).suffix.lower()}:{h}"
    with _LOCK:
        _STAT[p] = (st.st_mtime_ns, st.st_size, key, hashed_at)
    return key


def fingerprint(db) -> tuple:
    """Cheap signature of a Database's content, used to detect in-place mutation of a shared object."""
    try:
        return (tuple(sorted(db.phases)), tuple(sorted(str(e) for e in db.elements)), len(db.species),
                len(db._parameters), sum(hash(p["parameter"]) for p in db._parameters.all()),
                tuple(sorted(db.symbols)), sum(hash(e) for e in db.symbols.values()),
                sum(hash(str(sorted(ph.model_hints.items()))) for ph in db.phases.values()))
    except Exception:  # noqa: BLE001 - never let the guard break a calculation
        return (id(db), time.time())


def _read(path: str):
    from pycalphad import Database
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(path)
    _TIMES["parse_s"] += time.perf_counter() - t0
    return db


def get_db(path: str, fresh: bool = False):
    """Load a pycalphad Database (TDB or ChemSage/FactSage .dat), cached by file content.

    The returned object is shared: treat it as read-only.  ``fresh=True`` returns a private, uncached copy
    (use it when you intend to modify the database)."""
    p = str(Path(path).resolve())
    if fresh or not _ENABLED:
        return _read(p)
    key = file_key(p)
    hit = _DBS.get(key)
    if hit is not None:
        db, fp, _ = hit
        if fingerprint(db) == fp:
            return db
        _forget(key)                    # somebody modified the shared object: re-read from disk
    db = _read(p)
    with _LOCK:
        old = _DBS.d.get(key)
        _DBS.put(key, (db, fingerprint(db), p))
        live = {id(v[0]) for v in _DBS.d.values()}
        for i in [i for i in _OWNED if i not in live]:
            del _OWNED[i]
        if old is not None:
            _OWNED.pop(id(old[0]), None)
        _OWNED[id(db)] = key
    return db


def _forget(dbkey: str) -> None:
    with _LOCK:
        hit = _DBS.d.pop(dbkey, None)
        if hit is not None:
            _OWNED.pop(id(hit[0]), None)
        _MODELS.drop(lambda k: k[0] == dbkey)
        _FAMILIES.drop(lambda k: k[0] == dbkey)
        _FACTORIES.drop(lambda k: k[0] == dbkey)


def db_key(db) -> Optional[str]:
    """Content key of a Database loaded by get_db (None for any other / modified Database)."""
    if not _ENABLED:
        return None
    with _LOCK:
        key = _OWNED.get(id(db))
        hit = _DBS.d.get(key) if key else None
    if hit is None or hit[0] is not db:
        return None
    if fingerprint(db) != hit[1]:
        _forget(key)
        return None
    return key


# ---------------------------------------------------------------------------------------------------------
# models and phase record factories
# ---------------------------------------------------------------------------------------------------------
_ORIG: Dict[str, Any] = {}


def _orig(name: str):
    if not _ORIG:
        from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
        from pycalphad.core.utils import instantiate_models
        _ORIG.update(instantiate_models=instantiate_models, PhaseRecordFactory=PhaseRecordFactory)
    return _ORIG[name]


def _comps_key(db, comps) -> tuple:
    from pycalphad.core.utils import unpack_species
    return tuple(sorted(str(getattr(s, "name", s)) for s in unpack_species(db, comps)))


def _param_syms(parameters) -> Tuple[tuple, bool]:
    """(sorted parameter symbol names, whether numeric values were given)."""
    if not parameters:
        return (), False
    from pycalphad.core.utils import extract_parameters
    syms, vals = extract_parameters(parameters)
    return tuple(str(s) for s in syms), bool(len(syms))


def _cls_key(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}@{id(cls)}"


def cached_models(dbf, comps, phases, model=None, parameters=None, symbols_only=True):
    """Drop-in for pycalphad.core.utils.instantiate_models that reuses Model instances."""
    from pycalphad import Model
    plain = not symbols_only or (model is not None and not (isinstance(model, type) and issubclass(model, Model)))
    key0 = None if plain else db_key(dbf)
    if key0 is None:
        return _orig("instantiate_models")(dbf, comps, phases, model=model, parameters=parameters,
                                           symbols_only=symbols_only)
    from pycalphad.core.utils import extract_parameters
    cls = model or Model
    psyms, _ = _param_syms(parameters)
    sym_params = extract_parameters(parameters)[0] if parameters else []
    ck = _comps_key(dbf, comps)
    out = {}
    t0 = time.perf_counter()
    for name in phases:
        k = (key0, ck, str(name), _cls_key(cls), psyms)
        mod = _MODELS.get(k)
        if mod is None:
            mod = cls(dbf, comps, name, parameters=sym_params)
            mod._pcache_key = k             # identifies cache-built instances (see _model_ident)
            _MODELS.put(k, mod)
        out[name] = mod
    _TIMES["models_s"] += time.perf_counter() - t0
    return out


def _model_ident(dbf, comps, models) -> Optional[tuple]:
    """(dbkey, comps, cls, psyms) if every model is a cache-built instance of one family, else None."""
    key0 = db_key(dbf)
    if key0 is None or not models:
        return None
    ck = _comps_key(dbf, comps)
    fam = None
    for name, mod in models.items():
        k = getattr(mod, "_pcache_key", None)
        if k is None or k[0] != key0 or k[1] != ck or k[2] != str(name):
            return None
        f = (k[0], k[1], k[3], k[4])
        if fam is None:
            fam = f
        elif fam != f:
            return None
    return fam


def _shared_prf_class():
    cls = _ORIG.get("_SharedPRF")
    if cls is not None:
        return cls
    PRF = _orig("PhaseRecordFactory")

    class _SharedPRF(PRF):
        """PhaseRecordFactory whose compiled callables live in a shared per-family store."""

        def __init__(self, store: Dict, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._store = store

        def _shared(self, key, build):
            with _LOCK:
                if key in self._store:
                    return self._store[key]
            val = build()
            with _LOCK:
                return self._store.setdefault(key, val)

        def get_phase_constraints(self, phase_name):
            return self._shared(("cons", phase_name),
                                lambda: PRF.get_phase_constraints.__wrapped__(self, phase_name))

        def get_phase_formula_moles_element(self, phase_name, element_name, per_formula_unit=True):
            return self._shared(("moles", phase_name, element_name, per_formula_unit),
                                lambda: PRF.get_phase_formula_moles_element.__wrapped__(
                                    self, phase_name, element_name, per_formula_unit=per_formula_unit))

        def get_phase_property(self, phase_name, property_name, include_grad=True, include_hess=True):
            return self._shared(("prop", phase_name, property_name, include_grad, include_hess),
                                lambda: PRF.get_phase_property.__wrapped__(
                                    self, phase_name, property_name, include_grad=include_grad,
                                    include_hess=include_hess))

    _ORIG["_SharedPRF"] = _SharedPRF
    return _SharedPRF


def cached_prf(dbf, comps, state_variables, models, parameters=None):
    """Drop-in for PhaseRecordFactory(dbf, comps, state_variables, models, parameters=None)."""
    ident = _model_ident(dbf, comps, models) if isinstance(models, dict) else None
    psyms, has_vals = _param_syms(parameters)
    if ident is None or psyms != ident[3]:
        return _orig("PhaseRecordFactory")(dbf, comps, state_variables, models, parameters=parameters)
    from pycalphad.core.utils import get_state_variables
    sv = tuple(sorted(str(s) for s in get_state_variables(models=models, conds=state_variables)))
    fam_key = ident + (sv,)
    store = _FAMILIES.get(fam_key)
    if store is None:
        store = _FAMILIES.put(fam_key, {})
    SharedPRF = _shared_prf_class()
    if has_vals:        # parameter values live in the factory (and its PhaseRecords): never share the object
        return SharedPRF(store, dbf, comps, state_variables, models, parameters=parameters)
    fkey = fam_key + (tuple(sorted(models)), tuple(id(m) for _, m in sorted(models.items())))
    prf = _FACTORIES.get(fkey)
    if prf is None or prf._store is not store:
        prf = _FACTORIES.put(fkey, SharedPRF(store, dbf, comps, state_variables, dict(models),
                                             parameters=parameters))
    return prf


# ---------------------------------------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------------------------------------
_TARGETS = ["pycalphad.core.workspace", "pycalphad.core.calculate",
            "pycalphad.mapping.strategy.strategy_base", "scheil.simulate"]
_ACTIVE = {"depth": 0, "saved": []}


def _patch():
    import importlib
    _orig("PhaseRecordFactory")
    saved = []
    for modname in _TARGETS:
        try:
            m = importlib.import_module(modname)
        except ImportError:
            continue
        for name, new in (("instantiate_models", cached_models), ("PhaseRecordFactory", cached_prf)):
            if hasattr(m, name):
                saved.append((m, name, getattr(m, name)))
                setattr(m, name, new)
    return saved


@contextlib.contextmanager
def use():
    """Route pycalphad model / phase-record construction through the cache while the block runs."""
    if not _ENABLED:
        yield
        return
    with _LOCK:
        if _ACTIVE["depth"] == 0:
            _ACTIVE["saved"] = _patch()
        _ACTIVE["depth"] += 1
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE["depth"] -= 1
            if _ACTIVE["depth"] == 0:
                for m, name, old in reversed(_ACTIVE["saved"]):
                    setattr(m, name, old)
                _ACTIVE["saved"] = []


def cached(fn):
    """Decorator: run the function inside ``use()``."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with use():
            return fn(*a, **kw)
    return wrapper


# ---------------------------------------------------------------------------------------------------------
# management
# ---------------------------------------------------------------------------------------------------------
def info() -> Dict[str, Any]:
    with _LOCK:
        n_compiled = sum(len(s) for s in _FAMILIES.d.values())
        return {"enabled": _ENABLED,
                "databases": {"entries": len(_DBS.d), "hits": _DBS.hits, "misses": _DBS.misses,
                              "files": sorted({v[2] for v in _DBS.d.values()})},
                "models": {"entries": len(_MODELS.d), "hits": _MODELS.hits, "misses": _MODELS.misses},
                "compiled": {"families": len(_FAMILIES.d), "callables": n_compiled,
                             "factories": len(_FACTORIES.d), "factory_hits": _FACTORIES.hits,
                             "factory_misses": _FACTORIES.misses},
                "seconds_spent": {k: round(v, 2) for k, v in _TIMES.items()},
                "limits": dict(LIMITS)}


def clear() -> None:
    with _LOCK:
        for c in (_DBS, _MODELS, _FAMILIES, _FACTORIES):
            c.clear()
        _STAT.clear()
        _OWNED.clear()
        for k in _TIMES:
            _TIMES[k] = 0.0


def set_enabled(flag: bool) -> None:
    """Switch caching on/off for this process (off: every call builds everything from scratch)."""
    global _ENABLED
    _ENABLED = bool(flag)
    if not flag:
        clear()


def register(tb) -> None:
    reset = clear

    @tb.add("calc_cache", "Process-wide cache of parsed databases, phase models and compiled code (makes repeated "
            "calculations on the same system fast; invalidated automatically when a database file changes). "
            "Shows hit counts and sizes; clear=true empties it.", {"clear": {"type": "boolean"}}, [])
    def calc_cache(clear: bool = False):
        before = info()
        if clear:
            reset()
            return "cache cleared (was: " + tb.j({k: before[k] for k in ("databases", "models", "compiled")}) + ")", []
        return tb.j(before), []
