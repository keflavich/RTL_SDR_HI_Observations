"""
Run an airspy_rx integration

airspy_rx v1.0.5 23 April 2016
Usage:
-r <filename>: Receive data into file
-w Receive data into file with WAV header and automatic name
 This is for SDR# compatibility and may not work with other software
[-s serial_number_64bits]: Open device with specified 64bits serial number
[-p packing]: Set packing for samples,
 1=enabled(12bits packed), 0=disabled(default 16bits not packed)
[-f frequency_MHz]: Set frequency in MHz between [24, 1900] (default 900MHz)
[-a sample_rate]: Set sample rate
[-t sample_type]: Set sample type,
 0=FLOAT32_IQ, 1=FLOAT32_REAL, 2=INT16_IQ(default), 3=INT16_REAL, 4=U16_REAL, 5=RAW
[-b biast]: Set Bias Tee, 1=enabled, 0=disabled(default)
[-v vga_gain]: Set VGA/IF gain, 0-15 (default 5)
[-m mixer_gain]: Set Mixer gain, 0-15 (default 5)
[-l lna_gain]: Set LNA gain, 0-14 (default 1)
[-g linearity_gain]: Set linearity simplified gain, 0-21
[-h sensivity_gain]: Set sensitivity simplified gain, 0-21
[-n num_samples]: Number of samples to transfer (default is unlimited)
[-d]: Verbose mode
"""

import subprocess
import logging
from astropy import units as u
import numpy as np
import tqdm

hi_restfreq = 1420.405751786*u.MHz

type_to_dtype = {0: np.complex64, 1: np.float32, 2: np.int16, 3: np.int16, 4: np.uint16, 5: np.uint8}
# this is a guess, but reading in complex floats should read real then imaginary in order.
# FLOAT32_REAL should just be dropping half the data, effectively
# INT16_IQ is .... super weird, it's integer real, integer imaginary, flipping back and forth - there's no built-in reader for that
type_to_nchan_mult = {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 8}

def run_airspy_rx_integration(frequency=hi_restfreq.to(u.MHz).value,
                              samplerate=int(1e7),
                              sample_time_s=60,
                              type=0,
                              gain=20,
                              lna_gain=14,
                              vga_gain=15,
                              mixer_gain=15,
                              bias_tee=1,
                              in_memory=None,
                              output_filename="1420_integration.rx"):
    """
    Run an airspy_rx integration
    """
    if type in (2,3,4,5):
        raise NotImplementedError(f"Type {type} not implemented")

    n_samples = int(samplerate * sample_time_s)
    bytes_per_sample = {0: 8, 1:4, 2: 4, 3: 2, 4:2, 5: 1}[type]
    logging.info(f"Expected file size: {n_samples * bytes_per_sample / 1024**3:.2f} GB")

    if in_memory is None:
        # do it in-memory if the file is less than 2GB
        in_memory = n_samples * bytes_per_sample < 2*1024**3

    command = f"airspy_rx -r {output_filename} -f {frequency} -a {samplerate} -t {type} -n {n_samples} -h {gain} -l {lna_gain} -d -v {vga_gain} -m {mixer_gain} -b {bias_tee}"

    result = subprocess.run(command, shell=True, capture_output=True)
    print(result.stdout.decode("utf-8"))

    if result.returncode != 0:
        raise RuntimeError(f"airspy_rx failed with return code {result.returncode}")

    average_integration(output_filename, samplerate, type_to_dtype[type])

def average_integration(filename, nchan, dtype, in_memory=False):
    """
    Compute the power spectrum and average over time
    """

    pbar = tqdm.tqdm(desc="Averaging integration")

    if in_memory:
        data = (np.fromfile(fn, dtype=dtype)).reshape(-1, nchan)
        dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))
        meanpower = dataft.mean(axis=0)
    else:
        with open(filename, "rb") as fh:
            accum = np.zeros(nchan, dtype=dtype)
            n_samples = 0
            while True:
                pbar.update(1)
                data = np.fromfile(fh, dtype=dtype, count=nchan)
                if len(data) == nchan:
                    dataft = np.fft.fftshift(np.abs(np.fft.fft(data))**2)
                    accum += dataft
                    n_samples += 1
                elif len(data) == 0:
                    break
                else:
                    raise ValueError(f"Expected {nchan} samples, got {len(data)}")

            meanpower = accum / n_samples

    return meanpower


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = subprocess.run("airspy_rx", capture_output=True)
    print(result.stdout.decode("utf-8"))

    result = subprocess.check_call("airspy_rx")
    print(result)

    run_airspy_rx_integration(sample_time_s=6)