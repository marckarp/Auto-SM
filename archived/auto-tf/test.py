import subprocess
import shlex
import os

createDirectory = "mkdir test"
p1 = subprocess.call(createDirectory, shell=True)

copyInference = "cp inference.py test"
p2 = subprocess.call(copyInference, shell=True)
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()