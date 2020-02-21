import matplotlib.pyplot as plt 
import numpy as np 
import pdb

"""-
init
-"""
nside = 4
seeds = [10,20,30,40,50,60] 


"""------------
Stationary case
---------------"""
mode1 = 's' #'s' for stationary, 'o' for orbiting  

logz_s = np.array([])
logzerr_s = np.array([])

"""-------------------------------------------------------------------------------------------
filename format : /home/tommy/lisaenv/blip/Feb19/[mode]_n[nside]_s[seed#]/logz_rs[seed#]_[mode].txt
for nside=4, seed=10, mode=s: /home/tommy/lisaenv/blip/s_n4_s10/logz_rs10_s.txt
----------------------------------------------------------------------------------------------"""
for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/Feb19/" + mode1 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode1 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    logz_s = np.append(logz_s, float("{:.8e}".format(float(last_line))))
    # logz_s = np.append(logz_s, float(last_line))


"""----------------------------------------------------------------------------------------------
filename format : /home/tommy/lisaenv/blip/Feb19/[mode]_n[nside]_s[seed#]/logzerr_rs[seed#]_[mode].txt
for nside=4, seed=10, mode=s: /home/tommy/lisaenv/blip/s_n4_s10/logzerr_rs10_s.txt
-------------------------------------------------------------------------------------------------"""
for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/Feb19/" + mode1 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logzerr_rs" + str(seeds[x]) + "_" + mode1 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    logzerr_s = np.append(logzerr_s, float("{:.3e}".format(float(last_line))))
    # logzerr_s = np.append(logzerr_s, float(last_line))

"""----------
Orbiting case
-------------"""
mode2 = 'o'

logz_o = np.array([])
logzerr_o = np.array([])

for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/Feb19/" + mode2 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode2 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    logz_o = np.append(logz_o, float("{:.8e}".format(float(last_line))))
    # logz_o = np.append(logz_o, int(last_line))


for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/Feb19/" + mode2 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logzerr_rs" + str(seeds[x]) + "_" + mode2 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    logzerr_o = np.append(logzerr_o, float("{:.3e}".format(float(last_line))))
    # logzerr_o = np.append(logzerr_o, int(last_line))

# logz_o = np.array(logz_o, dtype=np.float128)
# logz_s = np.array(logz_s, dtype=np.float128)
# logzerr_o = np.array(logzerr_o, dtype=np.float128)
# logzerr_s = np.array(logzerr_s, dtype=np.float128)

"""
nside = 8 data
"""
nside2 = 8
logz_s2 = []
logz_o2 = []

for x in range(len(seeds)):
    files = open("/home/tommy/lisaenv/blip/Feb19/" + mode1 + "_n" + str(nside2) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode1 + ".txt")
    last_line = files.readlines()[-1].replace("\n",'')
    logz_s2 = np.append(logz_s2, float("{:.8e}".format(float(last_line))))
    
for x in range(len(seeds)):
    files = open("/home/tommy/lisaenv/blip/Feb19/" + mode2 + "_n" + str(nside2) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode2 + ".txt")
    last_line = files.readlines()[-1].replace("\n",'')
    logz_o2 = np.append(logz_o2, float("{:.8e}".format(float(last_line))))

"""-----
ANALYSIS
--------"""
log_bayes_factor = logz_o - logz_s
log_bayes_factor_8 = logz_o2 - logz_s2

plt.plot(seeds, log_bayes_factor, 'o', label='nside = 4')
plt.plot(seeds, log_bayes_factor_8, 'o', label='nside = 8')
# plt.errorbar(seeds, bayes_factor, yerr=bf_err, fmt='o')
plt.title(" Log Bayes' Factor of Orbiting vs Stationary for nside = " + str(nside) + " & nside = 8", fontsize=13)
plt.xlabel('Seeds #', fontsize=15)
# plt.ylabel(r"$\left|\log_{10} \ \frac{z_o}{z_s} \right|$ ", fontsize=15)
plt.ylabel(r"$\log_{10} \ \frac{z_o}{z_s} $ ", fontsize=15)
plt.legend()
plt.show()

"""
Data
"""
# out_file = open("bayesFactor_n4_n8.txt", 'w')
# print("nside = 4 logz_o", file=out_file)
# print(logz_o, file=out_file)
# print(file=out_file)
# print("nside = 4 logz_s", file=out_file)
# print(logz_s, file=out_file)
# print(file=out_file)
# print("nside = 4 logz_o - logz_s", file=out_file)
# print(log_bayes_factor, file=out_file)
# print(file=out_file)
# print(file=out_file)
# print("nside = 8 logz_o", file=out_file)
# print(logz_o2,file=out_file)
# print(file=out_file)
# print("nside = 8 logz_s", file=out_file)
# print(logz_s2,file=out_file)
# print(file=out_file)
# print("nside = 8 logz_o - logz_s", file=out_file)
# print(log_bayes_factor_8, file=out_file)



# bf_err = np.sqrt(zerr_o**2 + (zerr_s * bayes_factor)**2)
# # bf_err = np.array(np.sqrt(zerr_o**2 * z_s**-2 + zerr_s**2 * (z_o/z_s**2)**2), dtype=np.float128)