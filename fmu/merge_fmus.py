from os import listdir, walk, chdir
from os.path import join, basename, isdir
from shutil import copytree
from tempfile import TemporaryDirectory
from tkinter.filedialog import askopenfilenames, Tk, asksaveasfilename
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import re

from more_itertools import collapse


def get_fmu_key(path):
    """some identifier to derive a unique identifier from the fmu"""
    root = ET.parse(join(path, "modelDescription.xml")).getroot()
    s = ''  # root.attrib['guid']
    s += ET.tostring(root.find('ModelVariables')).decode('utf-8')
    s += ET.tostring(root.find('ModelStructure')).decode('utf-8')

    # replace all floats
    s = re.sub('"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?"', lambda f: str(float(f[0][1:-1])), s)
    return s


def copy_binaries(wd, fmus, binaries):
    primary = basename(fmus[0])
    for fmu, binaries_ in zip(fmus[1:], binaries[1:]):
        for binary in binaries_:
            copytree(join(wd, basename(fmu), 'binaries', binary), join(wd, primary, 'binaries', binary))


if __name__ == '__main__':
    tk = Tk()
    tk.withdraw()

    fmus = askopenfilenames(title='Select FMUs to merge', filetypes=['Functional\u00A0Mockup\u00A0Unit {*.fmu}'])

    if len(fmus) < 2:
        raise ValueError('Please select multiple FMUs')

    # create tmp directory
    with TemporaryDirectory() as dir:
        # unpack fmus to tmp directory
        binaries = []
        for fmu in fmus:
            with ZipFile(fmu, 'r') as z:
                z.extractall(join(dir, basename(fmu)))
            # add all subdirectories of "binaries" in the zip files
            binaries.append(list(filter(lambda f: not isdir(f), listdir(join(dir, basename(fmu), "binaries")))))

        # check if all fmus provide different binaries (binary/*)
        if len(set(collapse(binaries))) != len(list(collapse(binaries))):
            raise ValueError(f'The provided binary folders are not unique: {list(zip(fmus, binaries))}')

        # check if their xmls are sufficiently similar
        if len(set([get_fmu_key(join(dir, basename(fmu))) for fmu in fmus])) > 1:
            raise ValueError('The FMUs appear to be generated from different model files.' +
                             str([get_fmu_key(join(dir, basename(fmu))) for fmu in fmus]))

        # copy content of binary folder from secondary fmus to the folder of the primary
        copy_binaries(dir, fmus, binaries)

        # change the working directory into the primary fmu folder which we want to package
        chdir(join(dir, basename(fmus[0])))
        # zip the primary folder, delete tempfolder
        with ZipFile(asksaveasfilename(filetypes=['Functional\u00A0Mockup\u00A0Unit {*.fmu}']), 'w') as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in walk('.'):
                for filename in filenames:
                    # Add file to zip
                    zipObj.write(join(folderName, filename))
