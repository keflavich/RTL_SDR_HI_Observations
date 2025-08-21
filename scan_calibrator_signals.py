raise NotImplementedError("This script is deprecated in AST4723 in 2025.  Use run_sdrplay_integration.py instead.")
import numpy as np
from rtlsdr import RtlSdr
from astropy import units as u

noaa_freqs = [162.400, 162.425, 162.450, 162.475, 162.500, 162.525, 162.550]*u.MHz
# https://www.weather.gov/nwr/sites?site=WXJ60
#https://www.weather.gov/nwr/stations?State=FL
noaa_freq = 162.475*u.MHz

def calibrate_on_noaa(device_index=0, calibrator_freq=noaa_freq, bandwidth=1.0*u.MHz,
                      passes=20, max_offset=300, default_offset=325):
    """
    Calibrate the RTL-SDR using a known NOAA weather station.

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

    sdr = RtlSdr(device_index=device_index)
    try:
        sdr.sample_rate = bandwidth.to(u.Hz).value
        sdr.gain = 15
        numsamples = 2048*4
        sdr.center_freq = calibrator_freq.to(u.Hz).value

        assert sdr.get_freq_correction() == 0

        if default_offset != 0:
            sdr.set_freq_correction(default_offset)

        foff = sdr.get_freq_correction()

        pses = []

        frq = np.fft.fftfreq(numsamples)
        idx = np.argsort(frq)

        for ii in range(passes):
            sdr.center_freq = calibrator_freq.to(u.Hz).value

            samples = sdr.read_samples(numsamples)


            ps = np.abs(np.fft.fft(samples))**2
            # this seems to be an unneeded hack
            # (but it might help avoid a spike at 0-offset?)
            ps[0] = np.mean(ps)
            pses.append(ps[idx])

            print(".", end='')

        mean_ps = np.mean(pses, axis=0)

        frequency = u.Quantity(sdr.fc*(1-foff/1e6) + sdr.rs*frq[idx], u.Hz)
        print()
        print(f"sdr.fc={sdr.fc}, sdr.center_freq={sdr.center_freq}")

        cutout = ((frequency > calibrator_freq*(1-max_offset/1e6)) &
                  (frequency < calibrator_freq*(1+max_offset/1e6)))

        max_ind = np.argmax(mean_ps[cutout])
        meas_freq = frequency[cutout][max_ind]
        meas_offset = (meas_freq - calibrator_freq) / calibrator_freq


    finally:
        sdr.close()

    print(f"Measured frequency offset is {meas_offset.decompose().value*1e6} parts per million (ppm)")
    return frequency, mean_ps, meas_freq, meas_offset.decompose()
