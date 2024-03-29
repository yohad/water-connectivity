from pathlib import Path
from multiprocessing import Pool
from functools import partial
from utils import get_name_suffix

import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from tqdm.contrib.concurrent import process_map


def get_h5_data(filepath):
    """
    Get the biomass and flux data from an HDF5 file.
    """
    with h5.File(filepath, "r") as f:
        biomass = np.array(f["tasks"]["biomass"])
        sim_time = np.array(f["scales"]["sim_time"])
        flux = np.array(f["tasks"]["flux"])

    return (sim_time, biomass, flux)


def get_sim_time(data_entry):
    """
    Get the simulation time from a data entry.
    """
    return data_entry[0][0]


def get_all_data(dirpath: Path):
    """
    Get all the data from a directory.
    """
    filepaths = dirpath.glob("*.h5")
    raw_data = []
    for filepath in filepaths:
        raw_data.append(get_h5_data(filepath))

    raw_data = sorted(raw_data, key=get_sim_time)

    sim_time = np.concatenate([data[0] for data in raw_data], axis=0)
    biomass = np.concatenate([data[1] for data in raw_data])
    flux = np.concatenate([data[2] for data in raw_data])

    return (sim_time, biomass, flux)


def plot_frame(index, biomass, total_flux, sim_time, output, biomass_clims):
    """
    Plot a single frame of biomass and flux.
    """
    biomass = biomass[index]
    fig, axs = plt.subplots(1, 2)

    axs[1].set_title(f"Biomass {np.mean(biomass):.2f}")
    im = axs[1].imshow(
        np.rot90(biomass, -1),
        cmap="YlGn",
        clim=biomass_clims,
        origin="lower",
    )
    fig.colorbar(im, ax=axs[0])

    axs[0].set_title("Flux")
    axs[0].plot(sim_time, total_flux)
    axs[0].plot(sim_time[index], total_flux[index], "ro")

    fig.suptitle(f"$t={sim_time[index]:.2f}$")

    save_path = os.path.join(output, f"frame_{index:04d}.png")
    fig.savefig(save_path)
    plt.close(fig)


def plot_biomass_flux(dirpath: Path, output: Path):
    """
    Plot the biomass and flux from a directory of HDF5 files.
    """
    print("Getting data...")
    sim_time, biomass, flux = get_all_data(dirpath)

    total_flux = np.abs(flux[:, :, 0].sum(axis=1))

    print("Setting partial function...")
    partial_func = partial(
        plot_frame,
        biomass=biomass,
        total_flux=total_flux,
        sim_time=sim_time,
        output=output,
        biomass_clims=(0, np.max(biomass)),
    )

    process_map(partial_func, range(sim_time.shape[0]), chunksize=16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precipitation", type=float, default=1.1, dest="precipitation"
    )
    parser.add_argument("-m", "--slope", type=float, default=0.1, dest="slope")
    parser.add_argument("-o", "--output", type=Path, default="output")

    args = parser.parse_args()

    # Check if snapshots exist
    suffix = get_name_suffix(args.precipitation, args.slope)
    snapshots_dir = args.output / "snapshots" / f"snapshots{suffix}"
    assert snapshots_dir.exists(), f"{snapshots_dir} does not exist"

    frames_output = args.output / "frames" / f"frames{suffix}"
    frames_output.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {frames_output}")
    plot_biomass_flux(snapshots_dir, frames_output)

    video_output = args.output / "videos"
    video_output.mkdir(parents=True, exist_ok=True)
    input_pattern = str(frames_output / "*.png")
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            "40",
            "-pattern_type",
            "glob",
            "-y",
            "-i",
            input_pattern,
            video_output / f"video{suffix}.mp4",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
