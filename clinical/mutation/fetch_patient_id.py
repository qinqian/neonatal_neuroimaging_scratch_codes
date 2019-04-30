import pydicom, sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    z = pydicom.dcmread(sys.argv[1])

print(z.PatientID)

