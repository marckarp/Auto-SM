#for testing random code snippets during build

tf_versions = ["1.10.0", "1.11.0", "1.12.0", "1.13.0", "1.14.0", "1.15.0", "1.15.2", "1.15.3", 
        "1.15.4", "1.15.5", "1.4.1", "1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.0.1", "2.0.2",
        "2.0.3", "2.0.4", "2.1.0", "2.1.1", "2.1.2", "2.1.3", "2.2.0", "2.2.1", "2.2.2", "2.3.0", "2.3.1",
        "2.3.2", "2.4.1", "2.4.3", "2.5.1", "2.6.0", "1.10", "1.11", "1.12", "1.13", "1.14", "1.15", "1.4",
        "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6"]


import subprocess
sampStr = "testfolder"

#bashCommand = f"mkdir {sampStr}"
#p1 = subprocess.check_output(bashCommand, shell=True)
#print(p1)

"""
import subprocess
# file and directory listing
returned_text = subprocess.check_output("model.tar.gz", shell=True, universal_newlines=True)
print("dir command to list file and directory")
print(returned_text)
"""

#check status or existence of a file
import subprocess
proc = subprocess.Popen(['ls','-l'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
myset=set(proc.stdout)
print(myset)
#print(check_model_data)
#if "model.tar.gz" in check_model_data:
 #       print("found")
#rint("not found")