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

def determine_path(possible_users=['student', 'lab-admin', 'lab', 'Public', 'admina']):
    anaconda_path = os.path.split(sys.executable)[0]
    binpath = os.path.join(anaconda_path, 'Library', 'bin')
    bias_tee_path = os.path.join(binpath, 'rtl_biast')

    if os.path.exists(bias_tee_path):
        return binpath
    else:
        root = os.path.abspath(os.sep)
        for username in possible_users:
            for anacondapath in ('anaconda3', 'Anaconda3', 'anaconda', 'Anaconda'):
                binpath = os.path.join(root, 'Users', username, anacondapath, 'Library', 'bin')
                bias_tee_path = os.path.join(binpath, 'rtl_biast')
                if os.path.exists(bias_tee_path) or os.path.exists(bias_tee_path+".exe"):
                    return binpath

    raise IOError("rtl_biast wasn't found in any of the search directories!  "
                  "Maybe it's not installed?")

def bias_tee(device_index=0, bias_tee_timeout=2, skip_bias_tee=False, state=1):
    binpath = determine_path()
    bias_tee_path = os.path.join(binpath, 'rtl_biast')

    response = subprocess.Popen([bias_tee_path, '-d', str(device_index), '-b', str(state)])
    return_code = response.wait(timeout=bias_tee_timeout)

    if return_code != 0 and not skip_bias_tee:
        raise IOError("Failed to turn the bias tee (the thing that powers the low-noise amplifier (LNA)) on.  "
                      f"Error value was {response}")
    elif return_code != 0:
        response.kill()

    return return_code

def bias_tee_on(**kwargs):
    return bias_tee(state=1, **kwargs)

def bias_tee_off(**kwargs):
    return bias_tee(state=0, **kwargs)

def record_integration(altitude, azimuth, tint, observatory_longitude=-82.3,
                       observatory_latitude=29.6,
                       obs_type='',
                       freqcorr=60,
                       sleep_time_factor=2,
                       device_index=0,
                       verbose=False,
                       timeout_factor=2.1,
                       skip_bias_tee=False,
                       bias_tee_timeout=2,
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
    device_index : int
        The device index number.  Generally should be zero, but if you get
        errors saying the device is unresponsive (USB error 12, for example),
        try setting this to be 1.
    sleep_time_factor : int
        The amount of time to sleep in the case that the USB dongle is
        unresponsive is set to (tint) * (sleep_time_factor).  If this case
        comes up often, you may need to unplug the dongle and let it cool
    timeout_factor : float
        When running an integration, how much longer than the integration time
        should you wait before killing the task?  Usually 2.1x overhead is
        enough, but if you get a lot of timeout errors, try going up as
        high as 3.0x.
    bias_tee_timeout : float
        Amount of time to wait for the bias tee to turn on.  If you get bias
        tee timeout errors, increase this value.
    skip_bias_tee : bool
        Skip the bias tee steps?  Don't do this if you're taking real data
        of HI, but it can be useful for debugging.
    verbose : bool
        Should the integration command be verbose?
    """

    bias_tee_on(device_index=device_index, bias_tee_timeout=bias_tee_timeout,
                skip_bias_tee=skip_bias_tee)


    arguments = ['-i', str(tint),
                 '--do_fsw',
                 f'--obs_lon={observatory_longitude}',
                 f'--obs_lat={observatory_latitude}',
                 f'--altitude={altitude}',
                 f'--device_index={device_index}',
                 f'--azimuth={azimuth}',
                 f'--suffix={obs_type}',
                 f'--freqcorr={freqcorr}']
    if verbose:
        arguments.append('--verbose')

    # https://github.com/keflavich/1420SDR/blob/master/1420_psd.py
    if not os.path.exists('1420_psd.py'):
        get_1420psd()

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
        bias_tee_off(device_index=device_index,
                     bias_tee_timeout=bias_tee_timeout,
                     skip_bias_tee=skip_bias_tee)

        time.sleep(sleep_time)
        print(f"Returning control to the terminal at {datetime.datetime.now()}")

    if verbose:
        print(f"outputs={outs}, errors={errs}")

    return outs, errs
