from tkinter.filedialog import askopenfilenames

fmus = askopenfilenames(title='Select FMUs to merge', filetypes='*.fmu')

# first file will be considered as primary

# create tmp directory

# unpack fmus to tmp directory

# check if all fmus provide different binaries (binary/*)
# TODO raise ValueError

# check if their xmls are sufficiently similar
# TODO raise ValueError

# copy content of binary folder from secondary fmus to the folder of the primary

# zip the primary folder, delete tempfolder
