import subprocess, sys
import tqdm.notebook

tint = 60
altitude = 80
azimuth = 180
for ii in tqdm.notebook.tqdm(range(24*(3600//tint))):
    assert 0 == subprocess.call([r'C:\Users\gluner\Anaconda3\Library\bin\bias_tee_on.bat'])
    proc = subprocess.Popen([sys.executable,
                            '1420_psd.py', '-i', str(tint),
                             '--do_fsw', '--obs_lon=-82.3', '--obs_lat=29.6',
                             f'--altitude={altitude}',
                             f'--azimuth={azimuth}',
                             '--freqcorr=60'])
    time.sleep(tint)
    print(datetime.datetime.now())
    try:
        outs, errs = proc.communicate(timeout=tint)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print(f"Killed task at {datetime.datetime.now()}.   Turning off for 2 minutes.")
        print(f"outs={outs}, errs={errs}")
        assert 0 == subprocess.call([r'C:\Users\gluner\Anaconda3\Library\bin\bias_tee_off.bat'])
        time.sleep(120)
        print(f"Resuming integration at {datetime.datetime.now()}")
    assert 0 == subprocess.call([r'C:\Users\gluner\Anaconda3\Library\bin\bias_tee_on.bat'])
