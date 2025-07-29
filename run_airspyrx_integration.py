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

import pylab as pl
import subprocess
import logging
from astropy import units as u
import numpy as np
import tqdm
import time
import datetime
import scipy.signal
from astropy.time import Time
from astropy.table import Table
import os
from time import perf_counter
from astropy import constants

hi_restfreq = 1420.405751786 * u.MHz

type_to_dtype = {0: np.complex64, 1: np.float32, 2: np.int16, 3: np.int16, 4: np.uint16, 5: np.uint8}
# this is a guess, but reading in complex floats should read real then imaginary in order.
# FLOAT32_REAL should just be dropping half the data, effectively
# INT16_IQ is .... super weird, it's integer real, integer imaginary, flipping back and forth - there's no built-in reader for that
type_to_nchan_mult = {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 8}

logger = logging.getLogger('AirspyRx')

def run_airspy_rx_integration(ref_frequency=hi_restfreq,
                              fsw=True,
                              fsw_throw=2.25*u.MHz,
                              samplerate=int(1e7),
                              sample_time_s=60,
                              n_integrations=10,
                              type=0,
                              gain=8,
                              lna_gain=14,
                              vga_gain=15,
                              mixer_gain=15,
                              bias_tee=1,
                              in_memory=None,
                              output_filename="1420_integration.rx",
                              cleanup=True,
                              channel_width=1*u.km/u.s,
                              sleep_between_integrations=0.1,
                              extra_sample_buffer=1.01,
                              doplot=True,
                              retry_on_dropped_samples=True,
                              do_waterfall=False,
                              **kwargs
                             ):
    """
    Run an airspy_rx integration

    The sample time ``sample_time_s`` is the total time to integrate.
    It will be broken into n_integrations individual integrations that will
    be averaged together.
    Fewer integrations is more efficient, but more memory intensive.

    The channel width ``channel_width`` is the width of the channel in velocity.
    The 10 MHz of bandwidth (about 2100 km/s) will be split into channels of
    this width.

    fsw_throw is the difference in frequency between the two frequencies when
    doing frequency switching.
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
    for ii in range(n_integrations):
        output_filename_thisiter = f"{output_filename}_{ii}"
        t0 = perf_counter()

        if fsw:
            frequency_to_tune = ref_frequency + fsw_throw/2 * (-1 if ii % 2 == 1 else 1)
        else:
            frequency_to_tune = ref_frequency

        logging.debug(f"Tuning to {frequency_to_tune:0.3f} MHz")

        nsamples_requested = int(n_samples // n_integrations * extra_sample_buffer)

        command = f"airspy_rx -r {output_filename_thisiter} -f {frequency_to_tune.to(u.MHz).value:0.3f} -a {samplerate} -t {type} -n {nsamples_requested} -h {gain} -l {lna_gain} -d -v {vga_gain} -m {mixer_gain} -b {bias_tee}"

        isok = False

        while not isok:
            result = subprocess.run(command, shell=True, capture_output=True)
            #print(result.stdout.decode("utf-8"), result.stderr.decode("utf-8"))

            data = np.fromfile(output_filename_thisiter, dtype=type_to_dtype[type])
            # there are always some dropped samples; the requested number has some kinda built-in shortfall
            if (len(data) >= n_samples // n_integrations * 0.9) or not retry_on_dropped_samples:
                isok = True
            else:
                logger.warning(f"Expected >={n_samples // n_integrations} samples (requested {nsamples_requested}), got {len(data)}: dropped samples! took {perf_counter() - t0:.2f} seconds.  Retrying...")

        filenames.append(output_filename_thisiter)

        if result.returncode != 0:
            if os.path.exists(output_filename_thisiter):
                now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
                logger.log(25, f"{now} iteration {ii+1} of {n_integrations} of airspy_rx ended with return code {result.returncode} in {perf_counter() - t0:.2f} seconds.  nsamples_requested={nsamples_requested}.  ")
                logger.info(f"stdout+stderr: {result.stdout.decode('utf-8') + result.stderr.decode('utf-8')}")
                if result.returncode not in (0, 3221225477):
                    logger.error(f"return code {result.returncode} is not good")
            else:
                raise RuntimeError(f"{now} iteration {ii+1} of {n_integrations} of airspy_rx ended with return code {result.returncode}.  stdout+stderr: {result.stdout.decode('utf-8') + result.stderr.decode('utf-8')}")

        time.sleep(sleep_between_integrations)

    if fsw:
        meanpower1 = average_integration(filenames[::2], samplerate=samplerate, dtype=type_to_dtype[type])
        meanpower2 = average_integration(filenames[1::2], samplerate=samplerate, dtype=type_to_dtype[type])
    else:
        meanpower = average_integration(filenames, samplerate=samplerate, dtype=type_to_dtype[type])

    savename_fits = output_filename.replace(".rx", ".fits")
    assert savename_fits.endswith(".fits")
    samplerate = u.Quantity(samplerate, u.Hz)
    fsw_throw = u.Quantity(fsw_throw, u.Hz)
    ref_frequency = u.Quantity(ref_frequency, u.Hz)
    if fsw:
        # this is correct: frequency = (np.fft.fftshift(np.fft.fftfreq(data.shape[1])) * samplerate + rfrq).astype(np.float32)
        frequency_array1 = (np.fft.fftshift(np.fft.fftfreq(meanpower1.size)) * samplerate + (ref_frequency + fsw_throw/2)).to(u.MHz)
        frequency_array2 = (np.fft.fftshift(np.fft.fftfreq(meanpower2.size)) * samplerate + (ref_frequency - fsw_throw/2)).to(u.MHz)
        logging.info(f'frequency array 1 extrema = {frequency_array1.min():0.3f} , {frequency_array1.max():0.3f} ')
        logging.info(f'frequency array 2 extrema = {frequency_array2.min():0.3f} , {frequency_array2.max():0.3f} ')
        assert frequency_array1.min() > 0, f"frequency_array1.min()={frequency_array1.min()}"
        assert frequency_array2.min() > 0, f"frequency_array2.min()={frequency_array2.min()}"
        # more sanity checks
        assert frequency_array1.min() < ref_frequency
        assert frequency_array1.max() > ref_frequency
        assert frequency_array2.min() < ref_frequency
        assert frequency_array2.max() > ref_frequency

        save_fsw_integration(filename=savename_fits,
                             frequency1=frequency_array1,
                             frequency2=frequency_array2,
                             meanpower1=meanpower1,
                             meanpower2=meanpower2,
                             ref_frequency=ref_frequency,
                             **kwargs)
    else:
        frequency_array = (np.fft.fftshift(np.fft.fftfreq(meanpower.size)) * samplerate + ref_frequency).to(u.MHz)
        assert frequency_array.min() > 0, f"frequency_array.min()={frequency_array.min()}"
        save_integration(filename=savename_fits, frequency=frequency_array, meanpower=meanpower, **kwargs)

    if do_waterfall:
        waterfall_plot(filenames[0],
                       ref_frequency=u.Quantity(frequency_to_tune, u.MHz),
                       samplerate=samplerate, fsw_throw=fsw_throw,
                       dtype=type_to_dtype[type], channel_width=channel_width)

    if doplot:
        plot_table(savename_fits)

    if cleanup:
        for filename in filenames:
            os.remove(filename)


def do_calibration_run(fsw=False):
    n_integrations = 2 if fsw else 1

    n_cals = 3 * 4 * 2 * 4 * 5
    progress = tqdm.tqdm(desc="Calibration run", total=n_cals)

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    for lna_gain in range(0, 14, 5):
        for vga_gain in range(0, 16, 5):
            for bias_tee in (0, 1):
                for mixer_gain in range(0, 16, 5):
                    for gain in range(0, 21, 5):
                        progress.update(1)
                        run_airspy_rx_integration(sample_time_s=0.05,
                                                  samplerate=int(1e7),
                                                  output_filename=f"1420_integration_lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}_{now}.rx",
                                                  in_memory=False,
                                                  cleanup=False,
                                                  gain=gain,
                                                  lna_gain=lna_gain,
                                                  vga_gain=vga_gain,
                                                  mixer_gain=mixer_gain,
                                                  bias_tee=bias_tee,
                                                  extra_sample_buffer=1,
                                                  retry_on_dropped_samples=False,
                                                  n_integrations=n_integrations,
                                                  fsw=fsw,
                                                  do_waterfall=False,
                                                  doplot=False,
                                                  sleep_between_integrations=0,
                                                  )


def summarize_calibration_run(fsw=False, starttime=None, allow_missing=False, verbose=True):
    meanpower = {}

    for lna_gain in range(0, 16, 5):
        for vga_gain in range(0, 16, 5):
            for bias_tee in (0, 1):
                for mixer_gain in range(0, 16, 5):
                    for gain in range(0, 21, 5):
                        try:
                            tb = Table.read(f"1420_integration_lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}_{starttime}.fits")
                        except Exception as ex:
                            if allow_missing:
                                if verbose:
                                    print(f"{ex}: Missing file 1420_integration_lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}_{starttime}.fits")
                                continue
                            else:
                                raise ex
                        if fsw:
                            meanpower[lna_gain, vga_gain, bias_tee, mixer_gain, gain] = (tb['meanpower1'].mean(), tb['meanpower2'].mean())
                        else:
                            meanpower[lna_gain, vga_gain, bias_tee, mixer_gain, gain] = tb['spectrum'].mean()

    return meanpower


def plot_calibration_run(fsw=False, starttime=None, allow_missing=False, verbose=True):


    for gain in range(0, 21, 5):
        fig = pl.figure(figsize=(10, 10), dpi=300)
        fig.suptitle(f'gain = {gain}')
        axes = fig.subplots(2, 2)
        for vga_gain, plotind in zip(range(0, 16, 5), (1, 2, 3, 4)):
            for lna_gain, linestyle in zip(range(0, 14, 5), ('-', '--', ':', '-.')):
                for mixer_gain, color in zip(range(0, 16, 5), ('r', 'g', 'b', 'y', 'm', 'c')):
                    for bias_tee, linewidth in zip((0, 1), (0.5, 1.5)):
                        try:
                            tb = Table.read(f"1420_integration_lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}_{starttime}.fits")
                        except Exception as ex:
                            if allow_missing:
                                if verbose:
                                    print(f"{ex}: Missing file 1420_integration_lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}_{starttime}.fits")
                                continue
                            else:
                                raise ex

                        #color = (color[0], bias_tee, color[1])

                        # legend doesn't work unless you hack it like this to use pyplot
                        ax = axes.flat[plotind-1]
                        pl.sca(ax)
                        if fsw:
                            pl.semilogy(tb['frequency'], tb['meanpower1'], label=f'lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}', color=color,
                                    linewidth=linewidth, linestyle=linestyle, )
                        else:
                            pl.semilogy(tb['frequency'], tb['spectrum'], label=f'lna{lna_gain}_vga{vga_gain}_bias{bias_tee}_mixer{mixer_gain}_gain{gain}', color=color,
                                    linewidth=linewidth, linestyle=linestyle, )
                        ax.set_title(f'vga_gain = {vga_gain}')

        pl.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    pl.tight_layout()
    pl.savefig(f"1420_integration_calibration_{starttime}.png", bbox_inches='tight', dpi=300)

def plot_table(filename, ref_frequency=hi_restfreq):
    import pylab as pl
    pl.clf()
    tbl = Table.read(filename)
    if 'meanpower1' in tbl.colnames and 'meanpower2' in tbl.colnames:
        ax = pl.subplot(2, 1, 1)
        ax.plot(tbl['frequency1'], tbl['meanpower1'], label='meanpower1')
        ax.plot(tbl['frequency2'], tbl['meanpower2'], label='meanpower2')
        pl.xlabel(f"Frequency [MHz]")
        ax2 = pl.subplot(2, 1, 2)
        velo = (ref_frequency - u.Quantity(tbl['frequency1'], u.MHz)) / ref_frequency * constants.c
        ax2.plot(velo, tbl['spectrum'], label='meanpower1 - meanpower2')
        ax2.set_xlabel("Velocity (km/s)")
    else:
        ax = pl.gca()
        ax.plot(tbl['frequency'], tbl['spectrum'])
        ax.set_xlabel("Frequency [MHz]")

    pl.tight_layout()
    outfilename = filename.replace(".fits", "_spectrum.png")
    if not outfilename.endswith(".png"):
        outfilename += ".png"
    pl.savefig(outfilename, bbox_inches='tight')


def average_integration(filenames, dtype, in_memory=False,
                        channel_width=1*u.km/u.s,
                        samplerate=1e7, ref_frequency=1420*u.MHz):
    """
    Compute the power spectrum and average over time
    """

    pbar = tqdm.tqdm(desc="Averaging integration")

    nchan = int(((samplerate*u.Hz / ref_frequency * constants.c) / channel_width).decompose())

    if in_memory:
        data = np.concatenate([(np.fromfile(filename, dtype=dtype))
                                for filename in filenames])
        datasize = data.size - (data.size % nchan)
        data = data[:datasize].reshape(datasize//nchan, nchan)

        dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))
        meanpower = dataft.mean(axis=0)
    else:
        accum = np.zeros(nchan, dtype=dtype)
        n_samples = 0
        for filename in filenames:
            pbar.update(1)
            data = np.fromfile(filename, dtype=dtype)
            datasize = data.size - (data.size % nchan)
            nmeasurements = datasize // nchan
            data = data[:datasize].reshape(nmeasurements, nchan)

            dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))
            accum += dataft.sum(axis=0)
            n_samples += nmeasurements

        meanpower = accum / n_samples

    return np.abs(meanpower)


def waterfall_plot(filename, ref_frequency=1420*u.MHz, samplerate=1e7, fsw_throw=5e6, dtype=np.complex64, channel_width=1*u.km/u.s):
    import pylab as pl
    from astropy.visualization import simple_norm

    ref_frequency = u.Quantity(ref_frequency, u.Hz)
    fsw_throw = u.Quantity(fsw_throw, u.Hz)
    samplerate = u.Quantity(samplerate, u.Hz)

    nchan = int(((samplerate / ref_frequency * constants.c) / channel_width).decompose())

    data = np.fromfile(filename, dtype=dtype)
    datasize = data.size - (data.size % nchan)
    nmeasurements = datasize // nchan
    data = data[:datasize].reshape(nmeasurements, nchan)

    # fft along axis=1 means that's the frequency axis
    dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))

    rfrq = (ref_frequency + fsw_throw/2)
    frequency = (np.fft.fftshift(np.fft.fftfreq(data.shape[1])) * samplerate + rfrq).astype(np.float32)
    freqrange = (max(frequency) - min(frequency)).decompose().value
    freq0 = min(frequency).value

    pl.clf()
    ax = pl.gca()
    im = ax.imshow(dataft, norm=simple_norm(dataft, stretch='log'), origin='lower')
    aspect = data.shape[1] / data.shape[0]
    # logging.debug(f"aspect={aspect}")
    ax.set_aspect(aspect)
    #pl.xlabel("Frequency (MHz)")
    total_time = (data.size / samplerate).decompose().value
    yticks = ax.yaxis.get_ticklocs()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{x/max(yticks) * total_time:.2f} s' for x in yticks])
    ax.set_ylabel("Time")
    xticks = ax.xaxis.get_ticklocs()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{((x-min(xticks))/max(xticks) * freqrange + freq0)/1e6:.2f} MHz' for x in xticks],
                       rotation=30)
    ax.set_xlabel("Frequency (MHz)")
    pl.colorbar(mappable=im)

    outfilename = filename.replace(".rx", "_waterfall.png")
    if not outfilename.endswith(".png"):
        outfilename += ".png"
    pl.savefig(outfilename, bbox_inches='tight')


def save_fsw_integration(filename, frequency1, frequency2, meanpower1, meanpower2, decimate=False, ref_frequency=1420*u.MHz, **kwargs):

    assert np.all(frequency1 > 1.35*u.GHz)
    assert np.all(frequency1 < 1.5*u.GHz)
    # only ever use order=1, higher-order biases/shifts the signal very significantly (changes the frequency by >10 MHz)
    if decimate:
        tbl = Table({'spectrum': scipy.signal.decimate(meanpower1 - meanpower2, decimate, n=1),
                     'frequency1': u.Quantity(scipy.signal.decimate(frequency1, decimate, n=1), frequency1.unit),
                     'frequency2': u.Quantity(scipy.signal.decimate(frequency2, decimate, n=1), frequency2.unit),
                     'meanpower1': scipy.signal.decimate(meanpower1, decimate, n=1),
                     'meanpower2': scipy.signal.decimate(meanpower2, decimate, n=1)
                    })
    else:
        tbl = Table({'spectrum': meanpower1 - meanpower2,
                     'frequency1': frequency1,
                     'frequency2': frequency2,
                     'meanpower1': meanpower1,
                     'meanpower2': meanpower2
                    })
    assert tbl['frequency1'].quantity.max() > ref_frequency
    assert tbl['frequency1'].quantity.min() < ref_frequency
    tbl.meta['ref_frequency'] = ref_frequency
    save_tbl(tbl, filename=filename, **kwargs)


def save_integration(filename, frequency, meanpower, decimate=False, **kwargs):

    if decimate:
        tbl = Table({'spectrum': scipy.signal.decimate(meanpower, decimate, n=1),
                     'frequency': scipy.signal.decimate(frequency, decimate, n=1)})
    else:
        tbl = Table({'spectrum': meanpower, 'frequency': frequency})

    save_tbl(tbl, filename=filename, **kwargs)

def save_tbl(tbl, filename, obs_lat=None, obs_lon=None, elevation=None, altitude=None, azimuth=None, int_time=6):

    if obs_lat is None:
        try:
            obs_lat, obs_lon, elevation = whereami()
        except Exception as ex:
            logger.warning(f"Unable to determine where you are because {ex}.  Setting lon, lat, elev = 0,0,0")
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

    # now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    # run_airspy_rx_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=False, decimate=5)

    # now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    #run_airspy_rx_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=True, decimate=5)
