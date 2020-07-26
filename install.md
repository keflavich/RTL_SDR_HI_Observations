Set up a new computer with the software needed for RLT-SDR HI observations.  The RTL-SDR USB dongle and LNA are both needed to go through this setup and test the drivers.

1. Follow instructions at https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/ through step 13
   * I did not need step 2
   * step 13 is a verification step
2. Download & extract https://github.com/rtlsdrblog/rtl-sdr/releases/tag/v1.1 -> https://github.com/rtlsdrblog/rtl-sdr/releases/download/v1.1/bt_driver.zip
3. Download and install the x86 version of the 2010 release of vcredist from https://www.microsoft.com/en-us/download/details.aspx?id=26999
4. Plug in the rltsdr and connect the LNA
5. Run the `bias_tee_on` script
6. Verify that the LNA is on: the light should turn on
7. Install & run Jupyter using anaconda (python3 version)
 * requirements include:
    astroplan
    astropy>=4.0
    reproject
    matplotlib
    numpy
    tqdm
    astroquery
8. Retrieve & run this script: [install_rtlsdr.py](install_rtlsdr.py) from within a Jupyter environment
9. To verify this will work for students, set up a notebook and run:

```
    import requests
    response = requests.get("https://github.com/keflavich/1420SDR/raw/master/1420_psd.py")
    response.raise_for_status()
    with open("1420_psd.py", "w") as fh:
        fh.write(response.text)
```

and in a new cell, replacing <USERNAME> appropriately,

```
  import subprocess
  assert 0 == subprocess.call([r'C:\Users\<USERNAME>\Anaconda3\Library\bin\bias_tee_on.bat'])
  %run 1420_psd.py -i 5 --do_fsw --progressbar --verbose --doplot
  assert 0 == subprocess.call([r'C:\Users\<USERNAME>\Anaconda3\Library\bin\bias_tee_on.bat'])
```

The LNA light should stay illuminated and a plot should appear.  If you get a USB -12 error message, try changing the device ID (e.g., --device=1).  You can use `rtl_tcp` from powershell to see a list of devices
