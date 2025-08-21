"""
Run an sdrplay integration using SoapySDR
"""

import warnings
import pylab as pl
import subprocess
import logging
from astropy import units as u
import numpy as np
from tqdm.auto import tqdm
import time
import datetime
import scipy.signal
from astropy.time import Time
from astropy.table import Table
import os
from time import perf_counter
from astropy import constants
import SoapySDR
from SoapySDR import SOAPY_SDR_RX as RX
# CF32 = complex floats (complex64)
from SoapySDR import SOAPY_SDR_CF32 as CF32

hi_restfreq = 1420.405751786 * u.MHz

logger = logging.getLogger('SdrPlay')

type_to_dtype = {CF32: np.complex64}


def load_sdrplay_device(antenna='B'):
    try:
        sdr = SoapySDR.Device({'driver': 'sdrplay'})
    except RuntimeError:
        sdr = SoapySDR.Device()

    sdr.setAntenna(RX, 0, f"Antenna {antenna}")

    return sdr


def run_sdrplay_integration(ref_frequency=hi_restfreq,
                              obs_type='',
                              fsw=True,
                              fsw_throw=2.25*u.MHz,
                              samplerate=int(1e7),
                              sample_time_s=60,
                              n_integrations=2,
                              if_gain=59,
                              rf_gain=27,
                              bias_tee=True,
                              in_memory=None,
                              output_filename="1420_integration.rx",
                              cleanup=True,
                              channel_width=1*u.km/u.s,
                              sleep_between_integrations=0.0,
                              doplot=True,
                              retry_on_dropped_samples=True,
                              do_waterfall=False,
                              bandwidth='max',
                              channel=0,
                              verbose=False,
                              **kwargs
                             ):
    """
    Run an sdrplay integration

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
    if verbose:
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_INFO)
        logger.setLevel(logging.INFO)
    else:
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_ERROR)
        logger.setLevel(logging.ERROR)
    sdr = load_sdrplay_device()
    bandwidth_range = [x.maximum() for x in sdr.getBandwidthRange(RX, channel)]
    samplerate_range = [x.maximum() for x in sdr.getSampleRateRange(RX, channel)]
    if samplerate > max(samplerate_range):
        raise ValueError(f"Samplerate {samplerate} is too high for this device")
    if bandwidth == 'max':
        bandwidth = max(bandwidth_range)
    else:
        bandwidth = float(bandwidth)
        if bandwidth > max(bandwidth_range):
            raise ValueError(f"Bandwidth {bandwidth} is too high for this device")

    if fsw and n_integrations < 2:
        raise ValueError("n_integrations must be at least 2 for frequency switching.  Either increase n_integrations or set fsw=False")
    if fsw and n_integrations % 2 == 1:
        raise ValueError("n_integrations must be even for frequency switching.")

    nchan = int(((u.Quantity(bandwidth, u.Hz) / ref_frequency * constants.c) / channel_width).decompose())
    nchan = scipy.fftpack.next_fast_len(nchan)

    n_samples = int(samplerate * sample_time_s)
    # lop off remainder
    n_samples = n_samples - (n_samples % nchan)
    # hard-coded to match CF32
    bytes_per_sample = 8
    logging.info(f"Expected raw file size: {n_samples * bytes_per_sample / 1024**3:.2f} GB")

    if in_memory is None:
        # do it in-memory if the file is less than 2GB
        in_memory = n_samples * bytes_per_sample < (2 * 1024**3)

    if fsw:
        ref_frequency1 = ref_frequency + fsw_throw/2
        ref_frequency2 = ref_frequency - fsw_throw/2

    buff = np.zeros(n_samples, np.complex64)

    filenames = []
    for ii in tqdm(range(n_integrations), desc="Integrating"):
        output_filename_thisiter = f"{output_filename}_{ii}"

        if fsw:
            frequency_to_tune = ref_frequency1 if ii % 2 == 0 else ref_frequency2
        else:
            frequency_to_tune = ref_frequency

        logging.debug(f"Tuning to {frequency_to_tune:0.3f} MHz")

        sdr.writeSetting("biasT_ctrl", bias_tee)
        sdr.setGain(RX, channel, 'RFDR', rf_gain)
        sdr.setGain(RX, channel, 'IFDR', if_gain)

        sdr.setSampleRate(RX, channel, samplerate)
        sdr.setFrequency(RX, channel, frequency_to_tune.to(u.Hz).value)
        sdr.setBandwidth(RX, channel, bandwidth)

        rxStream = sdr.setupStream(RX, CF32)
        sdr.activateStream(rxStream) #start streaming

        sr = sdr.readStream(rxStream, [buff], len(buff))
        if verbose:
            print(f'return code: {sr.ret} flags: {sr.flags} timeNs: {sr.timeNs}')

        np.save(output_filename_thisiter, buff)

        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)

        filenames.append(output_filename_thisiter + '.npy')

        # reset buffer
        buff[:] = 0

        if sleep_between_integrations > 0:
            time.sleep(sleep_between_integrations)

    meta = {
        'obs_type': obs_type,
        'REFFREQ': ref_frequency.to(u.Hz).value,
        'RATE': samplerate,
        'CHANWID': (channel_width.value, channel_width.unit.to_string()),
        'N_INT': n_integrations,
        'if_gain': if_gain,
        'rf_gain': rf_gain,
        'bias_tee': bias_tee,
        'bandwid': bandwidth,
        'rx_chan': channel,
        'nchan': nchan,
        'nsamples': n_samples,
    }

    if fsw:
        # ref freq needs to be the same, otherwise the number of channels differs slightly
        meanpower1 = average_integration(filenames[::2], samplerate=samplerate, dtype=type_to_dtype[CF32], ref_frequency=ref_frequency, nchan=nchan)
        meanpower2 = average_integration(filenames[1::2], samplerate=samplerate, dtype=type_to_dtype[CF32], ref_frequency=ref_frequency, nchan=nchan)
        meta['fswthrow'] = fsw_throw.to(u.Hz).value
        meta['reffreq1'] = ref_frequency1.to(u.Hz).value
        meta['reffreq2'] = ref_frequency2.to(u.Hz).value
    else:
        meanpower = average_integration(filenames, samplerate=samplerate, dtype=type_to_dtype[CF32], ref_frequency=ref_frequency, nchan=nchan)

    savename_fits = output_filename.replace(".rx", ".fits")
    assert savename_fits.endswith(".fits")
    samplerate = u.Quantity(samplerate, u.Hz)
    fsw_throw = u.Quantity(fsw_throw, u.Hz)
    ref_frequency = u.Quantity(ref_frequency, u.Hz)
    if fsw:
        # this is correct: frequency = (np.fft.fftshift(np.fft.fftfreq(data.shape[1])) * samplerate + rfrq).astype(np.float32)
        frequency_array1 = (np.fft.fftshift(np.fft.fftfreq(meanpower1.size)) * samplerate + ref_frequency1).to(u.MHz)
        frequency_array2 = (np.fft.fftshift(np.fft.fftfreq(meanpower2.size)) * samplerate + ref_frequency2).to(u.MHz)
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
                             fsw_throw=fsw_throw,
                             meta=meta,
                             **kwargs)
    else:
        frequency_array = (np.fft.fftshift(np.fft.fftfreq(meanpower.size)) * samplerate + ref_frequency).to(u.MHz)
        assert frequency_array.min() > 0, f"frequency_array.min()={frequency_array.min()}"
        save_integration(filename=savename_fits, frequency=frequency_array, meanpower=meanpower, ref_frequency=ref_frequency, meta=meta, **kwargs)

    if do_waterfall:
        waterfall_plot(filenames[0],
                       ref_frequency=u.Quantity(frequency_to_tune, u.MHz),
                       samplerate=samplerate, fsw_throw=fsw_throw,
                       dtype=type_to_dtype[CF32], channel_width=channel_width)

    if doplot:
        plot_table(savename_fits)

    if cleanup:
        for filename in filenames:
            os.remove(filename)


def plot_table(filename, ref_frequency=hi_restfreq):
    import pylab as pl
    pl.clf()
    tbl = Table.read(filename)
    if 'power1' in tbl.colnames and 'power2' in tbl.colnames:
        ax = pl.subplot(2, 1, 1)
        ax.plot(tbl['frequency1'], tbl['power1'], label='power1')
        ax.plot(tbl['frequency2'], tbl['power2'], label='power2')
        pl.xlabel(f"Frequency [MHz]")
        ax2 = pl.subplot(2, 1, 2)
        velo = (ref_frequency - u.Quantity(tbl['frequency1'], u.MHz)) / ref_frequency * constants.c
        ax2.plot(velo, tbl['fsw_spectrum'], label='fsw_spectrum')
        ax2.set_xlabel("Velocity (km/s)")
    else:
        ax = pl.gca()
        ax.plot(tbl['frequency'], tbl['power'])
        ax.set_xlabel("Frequency [MHz]")

    pl.tight_layout()
    outfilename = filename.replace(".fits", "_spectrum.png")
    if not outfilename.endswith(".png"):
        outfilename += ".png"
    pl.savefig(outfilename, bbox_inches='tight')


def average_integration(filenames, dtype, in_memory=False,
                        channel_width=1*u.km/u.s,
                        nchan=None,
                        samplerate=1e7, ref_frequency=hi_restfreq):
    """
    Compute the power spectrum and average over time
    """

    pbar = tqdm(desc="Averaging integration")

    if in_memory:
        data = np.concatenate([(np.load(filename))
                                for filename in filenames])
        datasize = data.size - (data.size % nchan)
        data = data[:datasize].reshape(datasize//nchan, nchan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore overflow warnings
            dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))
        meanpower = dataft.mean(axis=0)
    else:
        accum = np.zeros(nchan, dtype=dtype)
        n_samples = 0
        for filename in filenames:
            pbar.update(1)
            data = np.load(filename)
            datasize = data.size - (data.size % nchan)
            nmeasurements = datasize // nchan

            # reshape such that each row is a single spectrum
            data = data[:datasize].reshape(nmeasurements, nchan)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore overflow warnings
                # FT along rows to produce a power spectrum
                dataft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=1))**2, axes=(1,))

            # sum across rows, then add to our accumulated sum spectrum
            accum += dataft.sum(axis=0)
            n_samples += nmeasurements

        assert n_samples > 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            meanpower = accum / n_samples

    return np.abs(meanpower)


def waterfall_plot(filename, ref_frequency=hi_restfreq, samplerate=1e7, fsw_throw=5e6, dtype=np.complex64, channel_width=1*u.km/u.s):
    import pylab as pl
    from astropy.visualization import simple_norm

    ref_frequency = u.Quantity(ref_frequency, u.Hz)
    fsw_throw = u.Quantity(fsw_throw, u.Hz)
    samplerate = u.Quantity(samplerate, u.Hz)

    nchan = int(((samplerate / ref_frequency * constants.c) / channel_width).decompose())

    data = np.load(filename)
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


def save_fsw_integration(filename, frequency1, frequency2, meanpower1, meanpower2, decimate=False, ref_frequency=hi_restfreq, meta={}, **kwargs):

    # only ever use order=1, higher-order biases/shifts the signal very significantly (changes the frequency by >10 MHz)
    if decimate:
        tbl = Table({'fsw_spectrum': scipy.signal.decimate(meanpower1 - meanpower2, decimate, n=1),
                     'frequency1': u.Quantity(scipy.signal.decimate(frequency1, decimate, n=1), frequency1.unit),
                     'frequency2': u.Quantity(scipy.signal.decimate(frequency2, decimate, n=1), frequency2.unit),
                     'power1': scipy.signal.decimate(meanpower1, decimate, n=1),
                     'power2': scipy.signal.decimate(meanpower2, decimate, n=1)
                    })
    else:
        tbl = Table({'fsw_spectrum': meanpower1 - meanpower2,
                     'frequency1': frequency1,
                     'frequency2': frequency2,
                     'power1': meanpower1,
                     'power2': meanpower2
                    })
    assert tbl['frequency1'].quantity.max() > ref_frequency
    assert tbl['frequency1'].quantity.min() < ref_frequency
    tbl.meta['REFFREQ'] = ref_frequency.to(u.Hz).value
    tbl.meta.update(meta)
    save_tbl(tbl, filename=filename, **kwargs)


def save_integration(filename, frequency, meanpower, decimate=False, ref_frequency=hi_restfreq, meta={}, **kwargs):

    if decimate:
        tbl = Table({'power': scipy.signal.decimate(meanpower, decimate, n=1),
                     'frequency': scipy.signal.decimate(frequency, decimate, n=1)})
    else:
        tbl = Table({'power': meanpower, 'frequency': frequency})

    tbl.meta['REFFREQ'] = ref_frequency.to(u.Hz).value
    tbl.meta.update(meta)
    save_tbl(tbl, filename=filename, **kwargs)

def save_tbl(tbl, filename, obs_lat=None, obs_lon=None, elevation=None, altitude=None, azimuth=None, int_time=6, **kwargs):

    if obs_lat is None:
        try:
            obs_lat, obs_lon, elevation = whereami()
        except Exception as ex:
            logger.warning(f"Unable to determine where you are because {ex}.  Setting lon, lat, elev = 0,0,0")
            obs_lat, obs_lon, elevation = 0, 0, 0

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    now_ap = Time.now()

    tbl.meta['LATITUDE'] = obs_lat
    tbl.meta['LONGITUD'] = obs_lon
    tbl.meta['SITEELEV'] = elevation
    tbl.meta['altitude'] = (altitude, "Observed altitude (deg)")
    tbl.meta['azimuth'] = (azimuth, "Observed azimuth (deg)")

    tbl.meta['tint'] = int_time
    tbl.meta['date-obs'] = now
    tbl.meta['mjd-obs'] = now_ap.mjd
    tbl.meta['jd-obs'] = now_ap.jd

    for key, value in kwargs.items():
        if len(key) > 8:
            key = key.replace("_", "")[:8]
        if hasattr(value, 'to'):
            value = value.to(u.Hz).value
        if key != 'meta':
            tbl.meta[key] = value

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        #warnings.simplefilter("ignore", astropy.warnings.VerifyWarning)
        if os.path.exists(filename):
            print(f"File {filename} already exists.  Saving as {filename.replace('.fits', f'_{now}.fits')}")
            filename = filename.replace(".fits", f"_{now}.fits")
        tbl.write(filename)


def whereami():
    import requests
    import geocoder
    myloc = geocoder.ip('me')
    lat, lon = myloc.latlng
    query = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'
    resp = requests.get(query)
    return lat, lon, resp.json()['results'][0]['elevation']


import specutils
import specutils.fitting
import specutils.manipulation
from specutils.manipulation.extract_spectral_region import _subregion_to_edge_pixels
from specutils import SpectralRegion, Spectrum
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import sigma_clip

hi_restfreq = 1420.405751786*u.MHz

def read_and_baseline(filename, polyorder=3, sigma=8):
    tbl = Table.read(filename)
    meta = tbl.meta
    data = tbl['spectrum']
    #data = tbl['meanpower1']
    wcs = WCS()
    wcs.wcs.crval[0] = tbl['frequency1'][0]
    wcs.wcs.cdelt[0] = tbl['frequency1'][1] - tbl['frequency1'][0]
    wcs.wcs.crpix[0] = 1
    wcs.wcs.cunit[0] = 'MHz'
    wcs.wcs.ctype[0] = 'FREQ'
    wcs.wcs.restfrq = hi_restfreq.to(u.Hz).value

    spectrum = Spectrum(flux=u.Quantity(data), wcs=wcs, meta=meta, velocity_convention='radio')
    def v_to_f(v):
        return spectrum.frequency[np.argmin(np.abs(spectrum.velocity - v))]

    left_pix, right_pix = _subregion_to_edge_pixels(SpectralRegion(v_to_f(-900*u.km/u.s), v_to_f(900*u.km/u.s)).subregions[0], spectrum)
    spectrum = Spectrum(flux=spectrum.flux[left_pix:right_pix], wcs=spectrum.wcs[left_pix:right_pix], meta=spectrum.meta, velocity_convention='radio')

    # 3rd order polyfit (works much better than poly1d, which fails)
    mod = np.polyval(np.polyfit(np.arange(spectrum.shape[0]), spectrum.flux, polyorder), np.arange(spectrum.shape[0]))
    spectrum = Spectrum(spectrum.flux - mod, wcs=spectrum.wcs, meta=spectrum.meta, velocity_convention='radio')

    clipped_flux = sigma_clip(spectrum.flux, sigma=sigma, cenfunc='median', stdfunc='std', masked=False, axis=0)

    spectrum = Spectrum(flux=u.Quantity(clipped_flux), wcs=spectrum.wcs, meta=spectrum.meta, velocity_convention='radio')

    ok = np.isfinite(spectrum.flux)
    mod = np.polyval(np.polyfit(np.arange(spectrum.shape[0])[ok], spectrum.flux[ok], polyorder), np.arange(spectrum.shape[0]))
    spectrum = Spectrum(spectrum.flux - mod, wcs=spectrum.wcs, meta=spectrum.meta, velocity_convention='radio')

    return spectrum


noaa_freqs = [162.400, 162.425, 162.450, 162.475, 162.500, 162.525, 162.550]*u.MHz
# https://www.weather.gov/nwr/sites?site=WXJ60
#https://www.weather.gov/nwr/stations?State=FL
noaa_freq = 162.475*u.MHz

def calibrate_on_noaa(device_index=0, calibrator_freq=noaa_freq, bandwidth=1.0*u.MHz,
                      integrations_per_pass=50,
                      passes=20, max_offset=300, default_offset=0, plot=False, verbose=False):
    """
    Calibrate the SDRPlay using a known NOAA weather station.

    Parameters
    ----------
    device_index : int
        The USB ID of the device.  Usually 0, sometimes 1.
    calibrator_freq : Quantity, MHz
        The frequency of the weather station.  The active one in Gainesville is
        preselected.
    bandwidth : Quantity, MHz
        The bandwidth of the observation to obtain.  Generally limited to <2.4
        MHz.  Will be used to set the sample rate of the SDR.
    passes : int
        Number of passes (single-integrations) to average prior to measuring
        the offset
    max_offset : int
        The maximum offset to search for, in parts-per-million.  This is set
        to avoid possibly detecting other (RFI) signals in-band.  Generally
        you want this to be less than the default_offset to avoid the "total power"
        spike at the center.
    default_offset : int
        The default offset to use when measuring the frequency offset.  A
        nonzero offset is needed to avoid having the signal channel landing on
        the central DC channel, which generally does not have a good
        measurement
    """
    if verbose:
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_INFO)
        logger.setLevel(logging.INFO)
    else:
        SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_ERROR)
        logger.setLevel(logging.ERROR)


    sdr = load_sdrplay_device()

    numsamples = 2**15
    samplerate = 0.125e6*u.Hz

    sdr.setSampleRate(RX, device_index, samplerate.to(u.Hz).value)
    sdr.setFrequency(RX, device_index, calibrator_freq.to(u.Hz).value)
    sdr.setBandwidth(RX, device_index, bandwidth.to(u.Hz).value)


    pses = []

    frq = np.fft.fftfreq(numsamples)
    idx = np.argsort(frq)

    buffer = np.zeros(numsamples * integrations_per_pass, dtype=np.complex64)

    for ii in tqdm(range(passes), desc="Calibrating on NOAA"):
        sdr.setFrequency(RX, device_index, calibrator_freq.to(u.Hz).value)

        rxStream = sdr.setupStream(RX, CF32)
        sdr.activateStream(rxStream) #start streaming
        sr = sdr.readStream(rxStream, [buffer], len(buffer))
        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)

        ps = (np.abs(np.fft.fft(buffer.reshape(integrations_per_pass, numsamples), axis=1))**2).mean(axis=0)

        # this seems to be an unneeded hack
        # (but it might help avoid a spike at 0-offset?)
        #ps[0] = np.mean(ps)
        pses.append(ps[idx])

    pses = np.array(pses)
    mean_ps = np.mean(pses, axis=0)

    frequency = (np.fft.fftshift(frq) * samplerate + calibrator_freq).to(u.MHz)

    cutout = ((frequency > calibrator_freq*(1-max_offset/1e6)) &
              (frequency < calibrator_freq*(1+max_offset/1e6)))

    max_ind = np.argmax(mean_ps[cutout])
    meas_freq = frequency[cutout][max_ind]
    meas_offset = (meas_freq - calibrator_freq) / calibrator_freq

    print()
    print(f"Selected frequency={sdr.getFrequency(RX, device_index)}")
    print(f"Measured frequency={meas_freq}")
    print(f"Measured frequency offset is {meas_offset.decompose().value*1e6} parts per million (ppm)")

    if plot:
        pl.clf()
        pl.plot(frequency.value, mean_ps)
        pl.plot(frequency[cutout].value, mean_ps[cutout], color='k', linewidth=2)
        pl.plot(frequency[cutout].value, pses[:, cutout].T)
        pl.xlim(frequency[cutout].min().value, frequency[cutout].max().value)
        pl.show()

    return frequency, mean_ps, meas_freq, meas_offset.decompose()


def bias_tee_on(device_index=0, sleep_time=1):
    sdr = load_sdrplay_device()
    sdr.writeSetting("biasT_ctrl", True)

    rxStream = sdr.setupStream(RX, CF32)
    sdr.activateStream(rxStream) #start streaming
    time.sleep(sleep_time)
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)


def bias_tee_off(device_index=0):
    sdr = load_sdrplay_device()
    sdr.writeSetting("biasT_ctrl", False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    calibrate_on_noaa(plot=True)

    # now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    # run_sdrplay_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=False, decimate=False)

    # now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    #run_airspy_rx_integration(sample_time_s=2, output_filename=f"1420_integration_{now}.rx", in_memory=True, decimate=5)
