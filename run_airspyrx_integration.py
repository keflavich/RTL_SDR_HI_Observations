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
import datetime
import scipy.signal
from astropy.time import Time
from astropy.table import Table
import os
from time import perf_counter

hi_restfreq = 1420.405751786 * u.MHz

type_to_dtype = {0: np.complex64, 1: np.float32, 2: np.int16, 3: np.int16, 4: np.uint16, 5: np.uint8}
# this is a guess, but reading in complex floats should read real then imaginary in order.
# FLOAT32_REAL should just be dropping half the data, effectively
# INT16_IQ is .... super weird, it's integer real, integer imaginary, flipping back and forth - there's no built-in reader for that
type_to_nchan_mult = {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 8}


def run_airspy_rx_integration(frequency=hi_restfreq.to(u.MHz).value,
                              fsw=True,
                              fsw_throw=int(5e6),
                              samplerate=int(1e7),
                              sample_time_s=60,
                              type=0,
                              gain=20,
                              lna_gain=14,
                              vga_gain=15,
                              mixer_gain=15,
                              bias_tee=1,
                              in_memory=None,
                              output_filename="1420_integration.rx",
                              cleanup=True,
                              **kwargs
                             ):
    """
    Run an airspy_rx integration
    """
    if type in (2, 3, 4, 5):
        raise NotImplementedError(f"Type {type} not implemented")
    if samplerate not in (int(1e7), int(2.5e6)):
        raise NotImplementedError(f"Samplerate {samplerate} not supported")

    n_samples = int(samplerate * sample_time_s)
    bytes_per_sample = {0: 8, 1: 4, 2: 4, 3: 2, 4: 2, 5: 1}[type]
    logging.info(f"Expected file size: {n_samples * bytes_per_sample / 1024**3:.2f} GB")

    if in_memory is None:
        # do it in-memory if the file is less than 2GB
        in_memory = n_samples * bytes_per_sample < (2 * 1024**3)

    filenames = []
    for ii in range(sample_time_s):
        output_filename_thisiter = f"{output_filename}_{ii}"
        t0 = perf_counter()

        if fsw:
            frequency_to_tune = frequency + fsw_throw/1e6/2 * (-1 if ii % 2 == 1 else 1)
        else:
            frequency_to_tune = frequency

        command = f"airspy_rx -r {output_filename_thisiter} -f {frequency_to_tune:0.3f} -a {samplerate} -t {type} -n {int(samplerate * 1.1)} -h {gain} -l {lna_gain} -d -v {vga_gain} -m {mixer_gain} -b {bias_tee}"

        isok = False

        while not isok:
            result = subprocess.run(command, shell=True, capture_output=True)
            #print(result.stdout.decode("utf-8"), result.stderr.decode("utf-8"))

            data = np.fromfile(output_filename_thisiter, dtype=type_to_dtype[type])
            if len(data) >= samplerate:
                isok = True
            else:
                print(f"Expected >={samplerate} samples, got {len(data)}: dropped samples! took {perf_counter() - t0:.2f} seconds.  Retrying...")

        filenames.append(output_filename_thisiter)

        if result.returncode != 0:
            if os.path.exists(output_filename_thisiter):
                print(f"iteration {ii} of {sample_time_s} of airspy_rx ended with return code {result.returncode} in {perf_counter() - t0:.2f} seconds (expected value was 1s)")
            else:
                raise RuntimeError(f"iteration {ii} of {sample_time_s} of airspy_rx ended with return code {result.returncode}")

    if fsw:
        meanpower1 = average_integration(filenames[::2], samplerate, type_to_dtype[type])
        meanpower2 = average_integration(filenames[1::2], samplerate, type_to_dtype[type])
    else:
        meanpower = average_integration(filenames, samplerate, type_to_dtype[type])

    savename_fits = output_filename.replace(".rx", ".fits")
    assert savename_fits.endswith(".fits")
    if fsw:
        frequency_array1 = (np.fft.fftshift(np.fft.fftfreq(meanpower.size)) * samplerate + (frequency + fsw_throw/1e6/2)*1e6).astype(np.float32)
        frequency_array2 = (np.fft.fftshift(np.fft.fftfreq(meanpower.size)) * samplerate + (frequency - fsw_throw/1e6/2)*1e6).astype(np.float32)

        save_integration(savename_fits,
                         frequency1=frequency_array1,
                         frequency2=frequency_array2,
                         meanpower1=meanpower1,
                         meanpower2=meanpower2, **kwargs)
    else:
        frequency_array = (np.fft.fftshift(np.fft.fftfreq(meanpower.size)) * samplerate + frequency*1e6).astype(np.float32)
        save_integration(savename_fits, frequency_array, meanpower=meanpower, **kwargs)

    if cleanup:
        for filename in filenames:
            os.remove(filename)


def average_integration(filenames, nchan, dtype, fsw, in_memory=False, overwrite=True):
    """
    Compute the power spectrum and average over time
    """

    pbar = tqdm.tqdm(desc="Averaging integration")

    if in_memory:
        data = np.array([(np.fromfile(filename, dtype=dtype, count=nchan))
                         for filename in filenames])
        dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))
        meanpower = dataft.mean(axis=0)
    else:
        accum = np.zeros(nchan, dtype=dtype)
        n_samples = 0
        for filename in filenames:
            pbar.update(1)
            data = np.fromfile(filename, dtype=dtype, count=nchan)
            dataft = np.fft.fftshift(np.abs(np.fft.fft(data))**2)
            accum += dataft
            n_samples += 1

        meanpower = accum / n_samples

    return np.abs(meanpower)


def save_fsw_integration(filename, f1, f2, meanpower1, meanpower2, decimate=False, **kwargs):

    if decimate:
        tbl = Table({'spectrum': scipy.signal.decimate(meanpower1 - meanpower2, decimate),
                     'frequency1': scipy.signal.decimate(frequency1, decimate),
                     'frequency2': scipy.signal.decimate(frequency2, decimate),
                     'meanpower1': scipy.signal.decimate(meanpower1, decimate),
                     'meanpower2': scipy.signal.decimate(meanpower2, decimate)
                    })
    else:
        tbl = Table({'spectrum': meanpower1 - meanpower2,
                     'frequency1': frequency1,
                     'frequency2': frequency2,
                     'meanpower1': meanpower1,
                     'meanpower2': meanpower2
                    })
    save_tbl(tbl, **kwargs)


def save_integration(filename, frequency, meanpower, decimate=False, **kwargs):

    if decimate:
        tbl = Table({'spectrum': scipy.signal.decimate(meanpower, decimate),
                     'frequency': scipy.signal.decimate(frequency, decimate)})
    else:
        tbl = Table({'spectrum': meanpower, 'frequency': frequency})

    save_tbl(tbl, **kwargs)

def save_tbl(tbl, obs_lat=None, obs_lon=None, elevation=None, altitude=None, azimuth=None, int_time=6):

    if obs_lat is None:
        try:
            obs_lat, obs_lon, elevation = whereami()
        except Exception as ex:
            print(f"Unable to determine where you are because {ex}.  Setting lon, lat, elev = 0,0,0")
            obs_lat, obs_lon, elevation = 0, 0, 0

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    now_ap = Time.now()
    
    tbl.meta['obs_lat'] = obs_lat
    tbl.meta['obs_lon'] = obs_lon
    tbl.meta['altitude'] = altitude
    tbl.meta['elevation'] = elevation
    tbl.meta['azimuth'] = azimuth

    tbl.meta['tint'] = int_time
    tbl.meta['date-obs'] = now
    tbl.meta['mjd-obs'] = now_ap.mjd
    tbl.meta['jd-obs'] = now_ap.jd

    tbl.write(filename)


def whereami():
    import requests
    import geocoder
    myloc = geocoder.ip('me')
    lat, lon = myloc.latlng
    query = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'
    resp = requests.get(query)
    return lat, lon, resp.json()['results'][0]['elevation']


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    run_airspy_rx_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=False, decimate=5)

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    run_airspy_rx_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=True, decimate=5)
