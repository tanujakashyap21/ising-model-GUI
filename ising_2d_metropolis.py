import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ============================================================
# 2D Ising model on a square lattice using Metropolis sampling
# Periodic boundary conditions, J = 1, k_B = 1
# ============================================================


def initialize_lattice(L, start="random", rng=None):
    """Create an L x L spin lattice with values +1 or -1."""
    if rng is None:
        rng = np.random.default_rng()

    if start == "up":
        return np.ones((L, L), dtype=int)
    if start == "down":
        return -np.ones((L, L), dtype=int)
    return rng.choice([-1, 1], size=(L, L))


def total_energy(spins):
    """
    Total energy of the lattice:
    E = - sum_<ij> s_i s_j
    Each bond is counted once by summing right and down neighbors only.
    """
    right_neighbors = np.roll(spins, shift=-1, axis=1)
    down_neighbors = np.roll(spins, shift=-1, axis=0)
    return -np.sum(spins * (right_neighbors + down_neighbors))


def total_magnetization(spins):
    """Total magnetization M = sum_i s_i."""
    return np.sum(spins)


def spin_correlation_profile(spins):
    """
    Average two-point correlation along the lattice axes.
    Returns <s_i s_{i+r}> averaged over all sites and both axis directions.
    """
    L = spins.shape[0]
    max_r = L // 2
    profile = np.empty(max_r + 1, dtype=float)
    spins_float = spins.astype(float, copy=False)

    for r in range(max_r + 1):
        corr_x = np.mean(spins_float * np.roll(spins_float, -r, axis=1))
        corr_y = np.mean(spins_float * np.roll(spins_float, -r, axis=0))
        profile[r] = 0.5 * (corr_x + corr_y)

    return profile


def metropolis_sweep(spins, temperature, rng):
    """
    Perform one Monte Carlo sweep.
    One sweep = L*L attempted spin flips at random sites.

    Returns:
        accepted_fraction : fraction of accepted trial flips in this sweep
        energy_change     : total change in energy during the sweep
        magnet_change     : total change in magnetization during the sweep
    """
    L = spins.shape[0]
    n_sites = L * L
    accepted = 0
    dE_total = 0
    dM_total = 0

    for _ in range(n_sites):
        i = rng.integers(0, L)
        j = rng.integers(0, L)

        s = spins[i, j]
        neighbor_sum = (
            spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
        )

        delta_E = 2 * s * neighbor_sum

        if temperature <= 0.0:
            accept_flip = delta_E <= 0
        else:
            accept_flip = delta_E <= 0 or rng.random() < np.exp(-delta_E / temperature)

        if accept_flip:
            spins[i, j] = -s
            accepted += 1
            dE_total += delta_E
            dM_total += -2 * s

    accepted_fraction = accepted / n_sites
    return accepted_fraction, dE_total, dM_total


def run_single_temperature(
    L,
    temperature,
    equil_sweeps,
    meas_sweeps,
    rng,
    start="random",
    store_snapshots=False,
):
    """
    Run the simulation at one temperature.

    Workflow:
    1. Initialize lattice
    2. Equilibrate for equil_sweeps
    3. Measure for meas_sweeps

    Returns a dictionary containing sweep histories, averages, fluctuations,
    and the final lattice.
    """
    spins = initialize_lattice(L, start=start, rng=rng)

    energy = total_energy(spins)
    magnetization = total_magnetization(spins)
    n_sites = L * L

    total_sweeps = equil_sweeps + meas_sweeps

    energy_history = np.zeros(total_sweeps)
    magnet_history = np.zeros(total_sweeps)
    abs_magnet_history = np.zeros(total_sweeps)
    acceptance_history = np.zeros(total_sweeps)

    measurement_energies = []
    measurement_magnets = []
    saved_snapshots = []

    for sweep in range(total_sweeps):
        accepted_fraction, dE, dM = metropolis_sweep(spins, temperature, rng)
        energy += dE
        magnetization += dM

        energy_history[sweep] = energy / n_sites
        magnet_history[sweep] = magnetization / n_sites
        abs_magnet_history[sweep] = abs(magnetization) / n_sites
        acceptance_history[sweep] = accepted_fraction

        if sweep >= equil_sweeps:
            measurement_energies.append(energy)
            measurement_magnets.append(magnetization)

            if store_snapshots and sweep == total_sweeps - 1:
                saved_snapshots.append(spins.copy())

    measurement_energies = np.array(measurement_energies, dtype=float)
    measurement_magnets = np.array(measurement_magnets, dtype=float)

    mean_energy = np.mean(measurement_energies) / n_sites
    mean_magnet = np.mean(measurement_magnets) / n_sites
    mean_abs_magnet = np.mean(np.abs(measurement_magnets)) / n_sites

    mean_energy_sq = np.mean(measurement_energies**2)
    mean_magnet_sq = np.mean(measurement_magnets**2)

    if temperature <= 0.0:
        heat_capacity = np.nan
        susceptibility = np.nan
    else:
        heat_capacity = (mean_energy_sq - np.mean(measurement_energies) ** 2) / (
            n_sites * temperature**2
        )
        susceptibility = (mean_magnet_sq - np.mean(measurement_magnets) ** 2) / (
            n_sites * temperature
        )

    return {
        "temperature": temperature,
        "final_lattice": spins.copy(),
        "energy_history": energy_history,
        "magnet_history": magnet_history,
        "abs_magnet_history": abs_magnet_history,
        "acceptance_history": acceptance_history,
        "mean_energy": mean_energy,
        "mean_magnet": mean_magnet,
        "mean_abs_magnet": mean_abs_magnet,
        "heat_capacity": heat_capacity,
        "susceptibility": susceptibility,
        "measurement_energies": measurement_energies / n_sites,
        "measurement_magnets": measurement_magnets / n_sites,
        "saved_snapshots": saved_snapshots,
    }


def scan_single_temperature(
    L,
    temperature,
    equil_sweeps,
    meas_sweeps,
    rng,
    start="up",
    store_correlation=False,
):
    """
    Lightweight single-temperature run used by temperature scans.
    Computes averages with running sums instead of storing full histories.
    """
    spins = initialize_lattice(L, start=start, rng=rng)
    energy = float(total_energy(spins))
    magnetization = float(total_magnetization(spins))
    n_sites = L * L

    energy_sum = 0.0
    energy_sq_sum = 0.0
    magnet_sum = 0.0
    magnet_abs_sum = 0.0
    magnet_sq_sum = 0.0
    correlation_sum = np.zeros(L // 2 + 1, dtype=float) if store_correlation else None

    total_sweeps = equil_sweeps + meas_sweeps

    for sweep in range(total_sweeps):
        _, dE, dM = metropolis_sweep(spins, temperature, rng)
        energy += dE
        magnetization += dM

        if sweep >= equil_sweeps:
            energy_sum += energy
            energy_sq_sum += energy * energy
            magnet_sum += magnetization
            magnet_abs_sum += abs(magnetization)
            magnet_sq_sum += magnetization * magnetization
            if store_correlation:
                correlation_sum += spin_correlation_profile(spins)

    mean_energy = energy_sum / (meas_sweeps * n_sites)
    mean_magnet = magnet_sum / (meas_sweeps * n_sites)
    mean_abs_magnet = magnet_abs_sum / (meas_sweeps * n_sites)

    if temperature <= 0.0:
        heat_capacity = np.nan
        susceptibility = np.nan
    else:
        mean_energy_sq = energy_sq_sum / meas_sweeps
        mean_magnet_sq = magnet_sq_sum / meas_sweeps
        mean_energy_raw = energy_sum / meas_sweeps
        mean_magnet_raw = magnet_sum / meas_sweeps
        heat_capacity = (mean_energy_sq - mean_energy_raw**2) / (n_sites * temperature**2)
        susceptibility = (mean_magnet_sq - mean_magnet_raw**2) / (n_sites * temperature)

    return {
        "temperature": temperature,
        "final_lattice": spins.copy(),
        "mean_energy": mean_energy,
        "mean_magnet": mean_magnet,
        "mean_abs_magnet": mean_abs_magnet,
        "heat_capacity": heat_capacity,
        "susceptibility": susceptibility,
        "measurement_count": meas_sweeps,
        "energy_sum": energy_sum,
        "energy_sq_sum": energy_sq_sum,
        "magnet_sum": magnet_sum,
        "magnet_abs_sum": magnet_abs_sum,
        "magnet_sq_sum": magnet_sq_sum,
        "correlation_sum": correlation_sum,
    }


def choose_snapshot_temperatures(temperatures, tc=2.269):
    """Pick low, near-critical, and high temperature values from the grid."""
    low_target = 1.5
    high_target = 3.5

    low_T = temperatures[np.argmin(np.abs(temperatures - low_target))]
    near_T = temperatures[np.argmin(np.abs(temperatures - tc))]
    high_T = temperatures[np.argmin(np.abs(temperatures - high_target))]
    return low_T, near_T, high_T


def build_temperature_grid(tc):
    """
    Use the original uniform 0.25 temperature grid.
    """
    del tc
    return np.arange(0.0, 4.5 + 0.25, 0.25)


def temperature_scan(L, temperatures, equil_sweeps, meas_sweeps, rng, start="up", n_replicas=1):
    """
    Run the Ising model over a temperature range and collect thermodynamic data.
    Also stores final lattices at low, near-critical, and high temperatures.
    """
    if n_replicas <= 0:
        raise ValueError("n_replicas must be at least 1.")

    mean_magnet = np.zeros_like(temperatures, dtype=float)
    mean_abs_magnet = np.zeros_like(temperatures, dtype=float)
    mean_energy = np.zeros_like(temperatures, dtype=float)
    heat_capacity = np.zeros_like(temperatures, dtype=float)
    susceptibility = np.zeros_like(temperatures, dtype=float)

    low_T, near_T, high_T = choose_snapshot_temperatures(temperatures)
    snapshot_data = {}
    correlation_data = {}

    for index, T in enumerate(temperatures):
        replica_results = []
        for replica_index in range(n_replicas):
            replica_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            result = scan_single_temperature(
                L=L,
                temperature=T,
                equil_sweeps=equil_sweeps,
                meas_sweeps=meas_sweeps,
                rng=replica_rng,
                start=start,
                store_correlation=(
                    np.isclose(T, low_T) or np.isclose(T, near_T) or np.isclose(T, high_T)
                ),
            )
            replica_results.append(result)

        total_measurements = sum(result["measurement_count"] for result in replica_results)
        energy_sum = sum(result["energy_sum"] for result in replica_results)
        energy_sq_sum = sum(result["energy_sq_sum"] for result in replica_results)
        magnet_sum = sum(result["magnet_sum"] for result in replica_results)
        magnet_abs_sum = sum(result["magnet_abs_sum"] for result in replica_results)
        magnet_sq_sum = sum(result["magnet_sq_sum"] for result in replica_results)

        mean_energy_raw = energy_sum / total_measurements
        mean_magnet_raw = magnet_sum / total_measurements
        mean_energy_sq = energy_sq_sum / total_measurements
        mean_magnet_sq = magnet_sq_sum / total_measurements

        mean_magnet[index] = mean_magnet_raw / (L * L)
        mean_abs_magnet[index] = magnet_abs_sum / (total_measurements * L * L)
        mean_energy[index] = mean_energy_raw / (L * L)

        if T <= 0.0:
            heat_capacity[index] = np.nan
            susceptibility[index] = np.nan
        else:
            heat_capacity[index] = (mean_energy_sq - mean_energy_raw**2) / (
                (L * L) * T**2
            )
            susceptibility[index] = (mean_magnet_sq - mean_magnet_raw**2) / (
                (L * L) * T
            )

        representative = replica_results[0]

        if np.isclose(T, low_T) or np.isclose(T, near_T) or np.isclose(T, high_T):
            snapshot_data[T] = representative["final_lattice"]
            correlation_sum = sum(result["correlation_sum"] for result in replica_results)
            correlation_data[T] = correlation_sum / total_measurements - mean_magnet[index] ** 2

        print(
            f"T = {T:.3f} | "
            f"<m> = {mean_magnet[index]:.4f} | "
            f"<|m|> = {mean_abs_magnet[index]:.4f} | "
            f"<e> = {mean_energy[index]:.4f} | "
            f"C = {heat_capacity[index]:.4f} | "
            f"chi = {susceptibility[index]:.4f}"
        )

    return {
        "temperatures": temperatures,
        "mean_magnet": mean_magnet,
        "mean_abs_magnet": mean_abs_magnet,
        "mean_energy": mean_energy,
        "heat_capacity": heat_capacity,
        "susceptibility": susceptibility,
        "snapshot_data": snapshot_data,
        "correlation_data": correlation_data,
        "snapshot_temperatures": (low_T, near_T, high_T),
    }


def plot_sweep_diagnostics(result, equil_sweeps, filename):
    """Plot magnetization, energy, and acceptance ratio versus sweep."""
    sweeps = np.arange(1, len(result["energy_history"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(sweeps, result["magnet_history"], color="tab:blue", lw=1.2)
    axes[0].axvline(equil_sweeps, color="black", ls="--", lw=1, label="equilibration end")
    axes[0].set_ylabel("Magnetization / spin")
    axes[0].set_title(
        f"Metropolis sweep history at T = {result['temperature']:.3f}"
    )
    axes[0].legend()

    axes[1].plot(sweeps, result["energy_history"], color="tab:red", lw=1.2)
    axes[1].axvline(equil_sweeps, color="black", ls="--", lw=1)
    axes[1].set_ylabel("Energy / spin")

    axes[2].plot(sweeps, result["acceptance_history"], color="tab:green", lw=1.2)
    axes[2].axvline(equil_sweeps, color="black", ls="--", lw=1)
    axes[2].set_ylabel("Acceptance ratio")
    axes[2].set_xlabel("Sweep")

    fig.tight_layout()
    fig.savefig(filename, dpi=200)


def plot_final_lattice(lattice, temperature, filename):
    """Plot the final lattice configuration."""
    plt.figure(figsize=(6, 6))
    plt.imshow(lattice, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Spin")
    plt.title(f"Final lattice configuration at T = {temperature:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)


def plot_thermodynamics(scan_data, tc, filename):
    """Plot temperature-dependent thermodynamic quantities."""
    T = scan_data["temperatures"]
    snapshot_temperatures = scan_data["snapshot_temperatures"]
    correlation_data = scan_data["correlation_data"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    plot_items = [
        (
            scan_data["mean_abs_magnet"],
            "Magnetization vs temperature",
            "<|m|> per spin",
        ),
        (scan_data["mean_energy"], "Average energy vs temperature", "<e> per spin"),
        (scan_data["heat_capacity"], "Heat capacity vs temperature", "C per spin"),
        (
            scan_data["susceptibility"],
            "Susceptibility vs temperature",
            "chi per spin",
        ),
    ]

    for ax, (y, title, ylabel) in zip(axes[:4], plot_items):
        ax.plot(T, y, "o-", ms=4)
        ax.axvline(tc, color="black", ls="--", lw=1, label=f"Tc = {tc:.3f}")
        ax.set_title(title)
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        ax.legend()

    corr_ax = axes[4]
    for label, temp in zip(["Low T", "Near Tc", "High T"], snapshot_temperatures):
        profile = correlation_data[temp]
        distances = np.arange(len(profile))
        corr_ax.plot(distances, profile, "o-", ms=4, label=f"{label}: T = {temp:.3f}")
    corr_ax.axhline(0.0, color="black", ls="--", lw=1)
    corr_ax.set_title("Connected spin correlation")
    corr_ax.set_xlabel("Separation r")
    corr_ax.set_ylabel("G(r)")
    corr_ax.legend()

    info_ax = axes[5]
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.98,
        "G(r) = <s_i s_{i+r}> - <s>^2\n"
        "shown for low, near-critical,\n"
        "and high temperatures.",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200)


def plot_temperature_snapshots(snapshot_data, snapshot_temperatures, filename):
    """Plot lattice snapshots at low, near-critical, and high temperature."""
    labels = ["Low T", "Near Tc", "High T"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, label, T in zip(axes, labels, snapshot_temperatures):
        lattice = snapshot_data[T]
        image = ax.imshow(
            lattice, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1
        )
        ax.set_title(f"{label}\nT = {T:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=axes, shrink=0.8, label="Spin")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)


def save_temperature_scan_csv(scan_data, filename):
    """Save thermodynamic observables versus temperature to a CSV file."""
    T = scan_data["temperatures"]
    data = np.column_stack(
        [
            T,
            scan_data["mean_magnet"],
            scan_data["mean_abs_magnet"],
            scan_data["mean_energy"],
            scan_data["heat_capacity"],
            scan_data["susceptibility"],
        ]
    )
    header = (
        "temperature,mean_magnet,mean_abs_magnet,"
        "mean_energy,heat_capacity,susceptibility"
    )
    np.savetxt(filename, data, delimiter=",", header=header, comments="")


def make_output_dir(base_dir="outputs"):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"ising_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_verification_report(scan_data, diagnostic_result, tc):
    """Generate a concise physics sanity-check report."""
    T = scan_data["temperatures"]
    abs_m = scan_data["mean_abs_magnet"]
    e = scan_data["mean_energy"]
    c = scan_data["heat_capacity"]
    chi = scan_data["susceptibility"]

    low_index = int(np.argmin(T))
    high_index = int(np.argmax(T))
    near_tc_index = int(np.argmin(np.abs(T - tc)))
    c_peak_index = int(np.argmax(c))
    chi_peak_index = int(np.argmax(chi))

    lines = [
        "2D Ising model verification report",
        f"Critical temperature Tc = {tc:.6f}",
        "",
        "Checks",
        (
            f"1. Low-temperature ordering: T = {T[low_index]:.3f}, "
            f"<|m|> = {abs_m[low_index]:.6f}"
        ),
        (
            f"2. High-temperature disorder: T = {T[high_index]:.3f}, "
            f"<|m|> = {abs_m[high_index]:.6f}"
        ),
        (
            f"3. Energy trend: e(T_low) = {e[low_index]:.6f}, "
            f"e(T_high) = {e[high_index]:.6f}"
        ),
        (
            f"4. Heat-capacity peak: T = {T[c_peak_index]:.3f}, "
            f"C = {c[c_peak_index]:.6f}"
        ),
        (
            f"5. Susceptibility peak: T = {T[chi_peak_index]:.3f}, "
            f"chi = {chi[chi_peak_index]:.6f}"
        ),
        (
            f"6. Diagnostic run at Tc: <m> = {diagnostic_result['mean_magnet']:.6f}, "
            f"<|m|> = {diagnostic_result['mean_abs_magnet']:.6f}, "
            f"<e> = {diagnostic_result['mean_energy']:.6f}"
        ),
        "",
        "Implementation notes",
        "7. Periodic boundaries are implemented with modular indexing and np.roll.",
        "8. One Monte Carlo sweep is defined as L*L attempted spin flips.",
        "",
        "Interpretation",
        (
            f"The scan is physically reasonable if low-T <|m|> is near 1, high-T <|m|> "
            f"is near 0, energy increases with temperature, and C and chi peak near Tc."
        ),
        (
            f"For this run, the temperature closest to Tc on the scan grid is "
            f"T = {T[near_tc_index]:.3f}."
        ),
    ]
    return "\n".join(lines) + "\n"


def save_verification_report(scan_data, diagnostic_result, tc, filename):
    """Write the verification report to disk."""
    with open(filename, "w", encoding="ascii") as file:
        file.write(build_verification_report(scan_data, diagnostic_result, tc))


def main():
    # ------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------
    L = 32
    equil_sweeps = 2000
    meas_sweeps = 4000
    n_replicas = 4
    seed = 12345

    # The exact critical temperature for the 2D square-lattice Ising model
    tc = 2.0 / np.log(1.0 + np.sqrt(2.0))

    # Representative run for sweep-by-sweep diagnostics
    diagnostic_temperature = tc

    # Temperature grid for thermodynamic averages
    temperatures = build_temperature_grid(tc)

    rng = np.random.default_rng(seed)
    output_dir = make_output_dir()

    print("=" * 60)
    print("2D Ising model with Metropolis sampling")
    print(f"Lattice size: {L} x {L}")
    print(f"Equilibration sweeps: {equil_sweeps}")
    print(f"Measurement sweeps:   {meas_sweeps}")
    print(f"Replica averages:     {n_replicas}")
    print(f"Critical temperature: Tc = {tc:.6f}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Single-temperature run for sweep diagnostics
    # ------------------------------------------------------------
    print("\nRunning sweep-by-sweep diagnostic simulation...")
    diagnostic_result = run_single_temperature(
        L=L,
        temperature=diagnostic_temperature,
        equil_sweeps=equil_sweeps,
        meas_sweeps=meas_sweeps,
        rng=rng,
        start="up",
    )

    plot_sweep_diagnostics(
        diagnostic_result,
        equil_sweeps=equil_sweeps,
        filename=output_dir / "sweep_diagnostics.png",
    )
    plot_final_lattice(
        diagnostic_result["final_lattice"],
        diagnostic_temperature,
        filename=output_dir / "final_lattice.png",
    )

    # ------------------------------------------------------------
    # Temperature scan for thermodynamic quantities
    # ------------------------------------------------------------
    print("\nRunning temperature scan...")
    scan_data = temperature_scan(
        L=L,
        temperatures=temperatures,
        equil_sweeps=equil_sweeps,
        meas_sweeps=meas_sweeps,
        rng=rng,
        start="up",
        n_replicas=n_replicas,
    )

    plot_thermodynamics(
        scan_data,
        tc=tc,
        filename=output_dir / "thermodynamic_quantities.png",
    )
    plot_temperature_snapshots(
        scan_data["snapshot_data"],
        scan_data["snapshot_temperatures"],
        filename=output_dir / "temperature_snapshots.png",
    )
    save_temperature_scan_csv(
        scan_data,
        filename=output_dir / "thermodynamic_data.csv",
    )
    save_verification_report(
        scan_data,
        diagnostic_result,
        tc,
        filename=output_dir / "verification_report.txt",
    )

    print(f"\nSaved outputs in: {output_dir}")
    print("\nSaved figures:")
    print(f"  {output_dir / 'sweep_diagnostics.png'}")
    print(f"  {output_dir / 'final_lattice.png'}")
    print(f"  {output_dir / 'thermodynamic_quantities.png'}")
    print(f"  {output_dir / 'temperature_snapshots.png'}")
    print("\nSaved data:")
    print(f"  {output_dir / 'thermodynamic_data.csv'}")
    print(f"  {output_dir / 'verification_report.txt'}")

    print("\nRepresentative averages at T = Tc:")
    print(f"  <m>   = {diagnostic_result['mean_magnet']:.6f}")
    print(f"  <|m|> = {diagnostic_result['mean_abs_magnet']:.6f}")
    print(f"  <e>   = {diagnostic_result['mean_energy']:.6f}")
    print(f"  C     = {diagnostic_result['heat_capacity']:.6f}")
    print(f"  chi   = {diagnostic_result['susceptibility']:.6f}")

    plt.show()


if __name__ == "__main__":
    main()
