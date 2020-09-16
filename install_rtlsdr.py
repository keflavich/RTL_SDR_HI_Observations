import pkgutil
from pathlib import Path
import sys
import requests
import zipfile
from io import BytesIO
from ctypes import CDLL
from ctypes.util import find_library
import subprocess
import os


subprocess.check_call([sys.executable, "-m", "pip", "install", 'https://github.com/roger-/pyrtlsdr/archive/master.zip'])

loader = pkgutil.get_loader('rtlsdr')
rtlsdr_path = Path(loader.path)
dllpath = rtlsdr_path.parents[2] / 'DLLs'
binpath = rtlsdr_path.parents[3] / 'Library' / 'bin'
print(f"dllpath={dllpath},  binpath={binpath}")

nbits = 64 if (sys.maxsize > 2**32) else 32
url = f'https://github.com/librtlsdr/librtlsdr/releases/download/v0.7.0/librtlsdr-0.7.0-13-gc79d-x{nbits}.zip'
response = requests.get(url)
response.raise_for_status()

zf = zipfile.ZipFile(BytesIO(response.content))
zf.extractall(binpath, members=[x for x in zf.namelist() if 'dll' in x])
zf.extractall(dllpath, members=[x for x in zf.namelist() if 'dll' in x])
print("Extracted ",[x for x in zf.namelist() if 'dll' in x])


response = requests.get('https://github.com/rtlsdrblog/rtl-sdr/releases/download/v1.1/bt_driver.zip')
response.raise_for_status()
zf = zipfile.ZipFile(BytesIO(response.content))
zf.extractall(binpath, members=[x for x in zf.namelist() if 'bias' in x])
tee_on = binpath / "bias_tee_on.bat"
assert os.path.isfile(tee_on)

assert find_library('librtlsdr'),"Installation failed."
assert find_library('librtlsdr.dll'),"Installation failed."
print("path to librtlsdr = {}".format(find_library('librtlsdr')))

assert subprocess.call([str(tee_on)]) == 0
print(tee_on)


"""
If the installation fails, check that both librtlsdr.dll and libusb-1.0.dll are on the Windows system executable path (not the python path).
Also, check that the dll is the appropriate 32 / 64 bit version
"""
