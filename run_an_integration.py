import time
import datetime
import subprocess
import sys
import os

def get_1420psd():
    print("Downloading the 1420psd.py script.  This will run only once, "
          "but requires an internet connection.  If you're not connected "
          "to the internet, either find the script and put it in the "
          f"current directory ({os.getcwd()}) or connect.")
    if not os.path.exists('1420psd.py'):
        import requests
        response = requests.get('https://raw.githubusercontent.com/keflavich/1420SDR/master/1420_psd.py')
        response.raise_for_status()
        with open('1420psd.py', 'w') as fh:
            fh.write(response.text)

def record_integration(altitude, azimuth, tint, observatory_longitude=-82.3,
                       observatory_latitude=29.6, sleep_time=120, username='student'):
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
    sleep_time : int, seconds
        The amount of time to sleep in the case that the USB dongle is
        unresponsive.  If this case comes up often, you may need to unplug the
        dongle and let it cool
    """

    response = subprocess.call([rf'C:\Users\{username}\Anaconda3\Library\bin\bias_tee_on.bat'])
    if response != 0:
        raise IOError("Failed to turn the bias tee (the thing that powers the low-noise amplifier (LNA)) on.  "
                      f"Error value was {response}")

    # https://github.com/keflavich/1420SDR/blob/master/1420_psd.py
    proc = subprocess.Popen([sys.executable,
                            '1420_psd.py', '-i', str(tint),
                             '--do_fsw',
                             f'--obs_lon={observatory_longitude}',
                             f'--obs_lat={observatory_latitude}',
                             f'--altitude={altitude}',
                             f'--azimuth={azimuth}',
                             '--freqcorr=60'])
    # wait for  the integration to complete
    time.sleep(tint)
    print(datetime.datetime.now())

    try:
        outs, errs = proc.communicate(timeout=tint)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print(f"The RTL-SDR failed to respond, so we're turning it off and waiting {sleep_time} seconds.")
        print(f"Killed task at {datetime.datetime.now()}.")
        print(f"outs={outs}, errs={errs}")
        response = subprocess.call([r'C:\Users\{username}\Anaconda3\Library\bin\bias_tee_off.bat'])
        if response != 0:
            raise IOError("Failed to turn the bias tee (the thing that powers the low-noise amplifier (LNA)) off.  "
                          f"Error value was {response}")

        time.sleep(sleep_time)
        print(f"Resuming integration at {datetime.datetime.now()}")
