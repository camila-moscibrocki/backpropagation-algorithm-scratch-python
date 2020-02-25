#!C:\Pycharm\lcc_rodas\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'seed==0.11.3','console_scripts','seed'
__requires__ = 'seed==0.11.3'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('seed==0.11.3', 'console_scripts', 'seed')()
    )
