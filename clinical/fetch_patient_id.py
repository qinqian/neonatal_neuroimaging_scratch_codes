import pydicom, sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    z = pydicom.dcmread(sys.argv[1])

if not hasattr(z, 'AcquisitionDate'):
    date = 'None'
else:
    date = z.AcquisitionDate
if not hasattr(z, 'SliceThickness'):
    thick = 'None'
else:
    thick = z.SliceThickness

if z.AccessionNumber.strip() == '':
    accession = 'None'
else:
    accession = z.AccessionNumber

if not hasattr(z, 'MagneticFieldStrength'):
    mag = 'None'
else:
    mag = str(z.MagneticFieldStrength)

print("\t".join([accession, z.PatientID, z.PatientBirthDate, date, z.PatientAge, z.StudyInstanceUID, str(thick), z.Manufacturer, z.ManufacturerModelName, mag, z.Modality]))
