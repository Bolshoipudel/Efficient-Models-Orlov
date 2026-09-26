import numpy as np

BYTES_PER_VALUE = 4
N_KERNELS = 25

FLOPS_PER_S2 = 17712
FLOPS_HEAD = 313344

N_WEIGHTS = 1045316
ACT_PER_S2 = 26
ACT_HEAD = 1124

BYTES_PER_S2 = 133
BYTES_HEAD = 2148


def flops(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return b * (FLOPS_PER_S2 * s**2 + FLOPS_HEAD)


def memory(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return BYTES_PER_VALUE * (N_WEIGHTS + b * (ACT_PER_S2 * s**2 + ACT_HEAD))


def bytes_moved(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return BYTES_PER_VALUE * (b * (BYTES_PER_S2 * s**2 + BYTES_HEAD) + N_WEIGHTS)


def latency(image_size, batch, theta):
    t_launch, p_eff, bw_eff = theta
    compute = flops(image_size, batch) / p_eff
    transfer = bytes_moved(image_size, batch) / bw_eff
    return N_KERNELS * t_launch + np.maximum(compute, transfer)


def energy(image_size, batch, theta_energy):
    p_idle, e_flop, e_byte, *theta = theta_energy
    return (
        p_idle * latency(image_size, batch, theta)
        + e_flop * flops(image_size, batch)
        + e_byte * bytes_moved(image_size, batch)
    )
