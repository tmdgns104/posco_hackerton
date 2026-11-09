import numpy as np

SCENARIOS = ("clean", "noise", "spike", "missing", "freeze")


def seed_for(group, phase, index, scenario):
    return int(np.random.SeedSequence([42, int(group), "EBS".index(phase),
                                       index, SCENARIOS.index(scenario)]).generate_state(1)[0])


def inject(window, scale, scenario, index, seed):
    rng = np.random.default_rng(seed)
    result = window.copy()
    if scenario == "noise":
        result += rng.normal(size=result.shape) * (0.2 * scale)
    elif scenario == "spike":
        selected = rng.choice(len(result), 16, replace=False)
        result[selected, 0] += rng.choice([-1, 1], 16) * 5 * scale[0]
    elif scenario == "missing" and index % 2 == 0:
        result[1280:1920] = np.nan
    elif scenario == "freeze" and index % 2 == 0:
        result[:, 0] = result[0, 0]
    elif scenario not in SCENARIOS:
        raise ValueError(scenario)
    return result


def inject_fault(window, fault_type, axis, scale, count, seed, train_scale, positions=None, signs=None):
    """Replayable parametric Gaussian noise and point impulses; the scale comes from training only."""
    if axis not in ("x","y","z","all") or scale < 0 or not np.isfinite(scale):
        raise ValueError("Invalid perturbation")
    std = np.asarray(train_scale,dtype=float)
    if std.shape!=(3,) or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError("Invalid training scale")
    rng = np.random.default_rng(seed)
    out = window.copy()
    cols = np.arange(3) if axis=="all" else np.array(["xyz".index(axis)])
    if fault_type=="noise":
        out[:,cols] += rng.normal(size=(len(out),len(cols))) * scale * std[cols]
        return out, [], []
    if fault_type!="spike" or not 0<count<=len(out):
        raise ValueError("Invalid impulse settings")
    pos = rng.choice(len(out),count,replace=False) if positions is None else np.asarray(positions,dtype=int)
    sign = rng.choice([-1,1],(count,len(cols))) if signs is None else np.asarray(signs,dtype=int)
    if (len(np.unique(pos))!=count or pos.shape!=(count,) or (pos<0).any() or (pos>=len(out)).any()
            or sign.shape!=(count,len(cols)) or not np.isin(sign,[-1,1]).all()):
        raise ValueError("Invalid replay data")
    out[np.ix_(pos,cols)] += sign * scale * std[cols]
    return out,pos.tolist(),sign.tolist()
