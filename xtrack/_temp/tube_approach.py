import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def load_knot_undulator(file_path):
    """
    Load a 3D magnetic field map from a text file and reshape into 3D arrays.

    Expected data columns:
        x, y, z, B_x, B_y, B_z

    Lines starting with '#' are ignored. The function infers the unique x/y/z
    coordinates and returns B-field components reshaped to (n, m, k), where:
        n = len(unique_x), m = len(unique_y), k = len(unique_z).
    """
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < 6:
        raise ValueError("Expected at least 6 columns: x y z B_x B_y B_z.")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    bx = data[:, 3]
    by = data[:, 4]
    bz = data[:, 5]

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)
    n, m, k = len(unique_x), len(unique_y), len(unique_z)

    expected_points = n * m * k
    if data.shape[0] != expected_points:
        raise ValueError(
            "Input points do not form a complete regular grid: "
            f"got {data.shape[0]} points, expected {expected_points} "
            f"from unique axis counts ({n}, {m}, {k})."
        )

    # Build index maps from coordinates to grid indices.
    ix = np.searchsorted(unique_x, x)
    iy = np.searchsorted(unique_y, y)
    iz = np.searchsorted(unique_z, z)

    if (
        np.any(unique_x[ix] != x)
        or np.any(unique_y[iy] != y)
        or np.any(unique_z[iz] != z)
    ):
        raise ValueError("Some coordinates do not match inferred axis values.")

    bx_3d = np.empty((n, m, k), dtype=bx.dtype)
    by_3d = np.empty((n, m, k), dtype=by.dtype)
    bz_3d = np.empty((n, m, k), dtype=bz.dtype)

    if np.unique(np.stack((ix, iy, iz), axis=1), axis=0).shape[0] != data.shape[0]:
        raise ValueError("Duplicate (x, y, z) points found in input data.")

    bx_3d[ix, iy, iz] = bx
    by_3d[ix, iy, iz] = by
    bz_3d[ix, iy, iz] = bz

    return unique_x, unique_y, unique_z, bx_3d, by_3d, bz_3d

def triangle(chi):
    """
    Triangle function T(chi), nonzero for chi in (-1, 1).
    """
    return np.maximum(1 - np.abs(chi), 0)

file_path = '/home/simonfan/projects/xsuite/xtrack/test_data/sls/undulator_field_map.txt'

mm_to_m = 1e-3

unique_x, unique_y, unique_z, bx_3d, by_3d, bz_3d = load_knot_undulator(file_path)

unique_x = unique_x * mm_to_m
unique_y = unique_y * mm_to_m
unique_z = unique_z * mm_to_m

n_planes = 400

# print(f"unique_x: {unique_x}")
# print(f"unique_y: {unique_y}")
# print(f"unique_z: {unique_z}")
# print(f"bx_3d: {bx_3d}")
# print(f"by_3d: {by_3d}")
# print(f"bz_3d: {bz_3d}")

# plt.plot(unique_z, bx_3d[0, 0, :])
# plt.plot(unique_z, by_3d[0, 0, :])
# plt.plot(unique_z, bz_3d[0, 0, :])
# plt.show()

# The highest multipole is the sextupole, which usually has order 2.
# But the highest power is 3.
order = 3

# Generate all (p,q) pairs with 0 < p+q <= M
pq_pairs = [(p, q) for p in range(order+1) 
                   for q in range(order+1) 
                   if 0 < p+q <= order]

n_coeffs = len(pq_pairs)

for p, q in pq_pairs:
    print(f"({p}, {q})")

# For bx equation: d/dx of x^p * y^q = p * x^(p-1) * y^q
def basis_bx(p, q, x, y):
    return p * x**(p-1) * y**q if p > 0 else 0

# For by equation: d/dy of x^p * y^q = q * x^p * y^(q-1)
def basis_by(p, q, x, y):
    return q * x**p * y**(q-1) if q > 0 else 0


def build_system_matrix(
    x_pts, y_pts, z_pts, bx_map, by_map, n_plane_pts, pq_pairs_local
):
    n_coeffs = len(pq_pairs_local)
    z_min, z_max = z_pts[0], z_pts[-1]
    planes_local = np.linspace(z_min, z_max, n_plane_pts)
    ds_local = planes_local[1] - planes_local[0]

    # Build sparse matrix and rhs.
    n_rows = 2 * len(x_pts) * len(y_pts) * len(z_pts)
    n_cols = n_plane_pts * n_coeffs
    A_local = lil_matrix((n_rows, n_cols))
    b_local = np.zeros(n_rows)

    row = 0
    for iz, z in enumerate(z_pts):
        for ix, x in enumerate(x_pts):
            for iy, y in enumerate(y_pts):
                # Triangle support makes this sparse in z.
                for j, z_j in enumerate(planes_local):
                    chi = (z - z_j) / ds_local
                    T = triangle(chi)
                    if T == 0:
                        continue

                    for idx, (p, q) in enumerate(pq_pairs_local):
                        col = j * n_coeffs + idx
                        A_local[row, col] += basis_bx(p, q, x, y) * T
                        A_local[row + 1, col] += basis_by(p, q, x, y) * T

                b_local[row] = bx_map[ix, iy, iz]
                b_local[row + 1] = by_map[ix, iy, iz]
                row += 2

    return A_local, b_local, planes_local

# Optional solve step:
A, b, planes = build_system_matrix(
    unique_x, unique_y, unique_z, bx_3d, by_3d, n_planes, pq_pairs
)
ds = planes[1] - planes[0]

result = lsqr(A.tocsr(), b)
coeff_vector = result[0]

psi = coeff_vector.reshape(n_planes, len(pq_pairs))

print(f"psi.shape: {psi.shape}")

# for idx, (p, q) in enumerate(pq_pairs):
#     plt.plot(planes, psi[:, idx], label=f'Ψ({p},{q})')
# plt.xlabel('z [m]')
# plt.ylabel('Ψ')
# plt.legend()
# plt.show()

def evaluate_field(x, y, z, psi, planes, ds, pq_pairs):
    """
    Evaluate the reconstructed field at a point (x, y, z).
    """
    bx = 0.0
    by = 0.0
    
    for j, z_j in enumerate(planes):
        chi = (z - z_j) / ds
        T = triangle(chi)
        if T == 0:
            continue
        
        for idx, (p, q) in enumerate(pq_pairs):
            bx += psi[j, idx] * basis_bx(p, q, x, y) * T
            by += psi[j, idx] * basis_by(p, q, x, y) * T
    
    return bx, by


def plot_field_at_xy(x, y, z_pts, psi, planes, pq_pairs, bx_3d, by_3d, z_unique):
    ds = planes[1] - planes[0]
    bx_eval = np.zeros_like(z_pts)
    by_eval = np.zeros_like(z_pts)
    for iz, z in enumerate(z_pts):
        bx_eval[iz], by_eval[iz] = evaluate_field(x, y, z, psi, planes, ds, pq_pairs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(z_pts, bx_eval)
    ax1.plot(z_unique, bx_3d[0, 0, :])
    ax1.set_ylabel('Bx [T]')
    ax1.grid()
    ax2.plot(z_pts, by_eval)
    ax2.plot(z_unique, by_3d[0, 0, :])
    ax2.set_ylabel('By [T]')
    ax2.set_xlabel('z [m]')
    ax2.grid()
    plt.show()

psi = coeff_vector.reshape(n_planes, n_coeffs)

plot_field_at_xy(0, 0, unique_z, psi, planes, pq_pairs, bx_3d, by_3d, unique_z)