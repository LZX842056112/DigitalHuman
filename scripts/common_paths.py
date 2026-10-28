from pathlib import Path
import re


def linux_path(value):
    value = str(value)
    if re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value)
