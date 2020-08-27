import time
import datetime
import subprocess
import sys
import os

def get_1420psd(overwrite=False):
    print("Downloading the 1420psd.py script.  This will run only once, "
          "but requires an internet connection.  If you're not connected "
          "to the internet, either find the script and put it in the "
          f"current directory ({os.getcwd()}) or connect.")
    if overwrite or not os.path.exists('1420_psd.py'):
        import requests
        response = requests.get('https://raw.githubusercontent.com/keflavich/1420SDR/master/1420_psd.py')
        response.raise_for_status()
        with open('1420_psd.py', 'w') as fh:
            fh.write(response.text)

def record_integration(altitude, azimuth, tint, observatory_longitude=-82.3,
                       observatory_latitude=29.6,
                       obs_type='',
                       freqcorr=60,
                       sleep_time_factor=2,
                       #anaconda_path='C:\\ProgramData\\Anaconda3\\',
                       verbose=False,
                       timeout_factor=2.1,
                       skip_bias_tee=False,
                      ):
    """
    Record a single integration

    Parameters
    ----------
    altitude : float, deg
        The altitude the telescope is pointing at
    azimuth : float, deg
        The azimuth the telescope is pointing toward
    tint : int, seconds
        The integration time to run (how long are you recording)
    observatory_latitude : float
    observatory_longitude : float
        Coordinates of the observatory (default: Gainesville)
    obs_type : str
        The observation type.  This will be appended to the filename,
        so make sure it's descriptive enough and does not include any
        spaces or non-ascii characters.
    freqcorr : int
        The frequency correction factor.  This value needs to be calibrated
        for each individual RTL-SDR.  Default is 60, but may be wrong!
    sleep_time_factor : int
        The amount of time to sleep in the case that the USB dongle is
        unresponsive is set to (tint) * (sleep_time_factor).  If this case
        comes up often, you may need to unplug the dongle and let it cool
    timeout_factor : float
        When running an integration, how much longer than the integration time
        should you wait before killing the task?  Usually 2.1x overhead is
        enough, but if you get a lot of timeout errors, try going up as
        high as 3.0x.
    skip_bias_tee : bool
        Skip the bias tee steps?  Don't do this if you're taking real data
        of HI, but it can be useful for debugging.
    verbose : bool
        Should the integration command be verbose?
    """
    anaconda_path = os.path.split(sys.executable)[0]

    response = subprocess.call([rf'{anaconda_path}\Library\bin\bias_tee_on.bat'])
    if response != 0 and not skip_bias_tee:
        raise IOError("Failed to turn the bias tee (the thing that powers the low-noise amplifier (LNA)) on.  "
                      f"Error value was {response}")

    arguments = ['-i', str(tint),
                 '--do_fsw',
                 f'--obs_lon={observatory_longitude}',
                 f'--obs_lat={observatory_latitude}',
                 f'--altitude={altitude}',
                 f'--azimuth={azimuth}',
                 f'--suffix={obs_type}',
                 f'--freqcorr={freqcorr}']
    if verbose:
        arguments.append('--verbose')

    # https://github.com/keflavich/1420SDR/blob/master/1420_psd.py
    proc = subprocess.Popen([sys.executable, '1420_psd.py'] + arguments)
    # wait for  the integration to complete
    time.sleep(tint * timeout_factor)
    print(datetime.datetime.now())

    try:
        outs, errs = proc.communicate(timeout=tint)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        sleep_time = sleep_time_factor * tint
        print(f"The RTL-SDR failed to respond, so we're turning it off and waiting {sleep_time} seconds.")
        print(f"Killed task at {datetime.datetime.now()}.")
        print(f"outs={outs}, errs={errs}")
        response = subprocess.call([rf'{anaconda_path}\Library\bin\bias_tee_off.bat'])
        if response != 0 and not skip_bias_tee:
            raise IOError("Failed to turn the bias tee (the thing that powers the low-noise amplifier (LNA)) off.  "
                          f"Error value was {response}")

        time.sleep(sleep_time)
        print(f"Returning control to the terminal at {datetime.datetime.now()}")

    if verbose:
        print(f"outputs={outs}, errors={errs}")

    return outs, errs
