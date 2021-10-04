import os

allFiles = list()
directory = 'Desktop/Thesis/all_wav/Converter/'

for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        filepath = os.path.join(directory, filename)
        allFiles.append(filepath)
# pitch, voicing and loudness
'''
for each in allFiles:
    name = each.replace('Desktop/Thesis/all_wav/Converter/','').replace('.wav','')
    os.system("SMILExtract -C ~/opensmile/config/prosody/prosodyAcf.conf -I " + each + " -O Desktop/all_wav_ftrs/prosody_"+name+".csv")
'''
# HNR
'''
for each in allFiles:
    name = each.replace('Desktop/Thesis/all_wav/Converter/','').replace('.wav','')
    os.system("SMILExtract -C ~/opensmile/config/prosody/prosodyShsAll.conf -I " + each + " -lld Desktop/all_hnr/hnr_"+name+".csv")
'''
