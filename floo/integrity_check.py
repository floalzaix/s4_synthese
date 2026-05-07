#
#   Imports
#

import hashlib as hlib

#
#   Constants
#

SHA256_FILE_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/SHA256SUMS.txt"

BASE_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/"

#
#   Main
#

with open(SHA256_FILE_PATH, "r") as file:
    for line in file:
        if line.startswith("#"):
            continue
        hash_value, file_path = line.split()
        with open(BASE_PATH + file_path, "rb") as f:
            data = f.read()
            hash_object = hlib.sha256(data)
            if hash_object.hexdigest() != hash_value:
                print(f"Hash mismatch for {file_path}")
            else:
                print(f"Hash matches for {file_path}")