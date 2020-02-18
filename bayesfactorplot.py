import matplotlib.pyplot as plt 
import numpy as np 
import pdb

"""-
init
-"""
nside = 4
seeds = [10] 


"""------------
Stationary case
---------------"""
mode1 = 's' #'s' for stationary, 'o' for orbiting  

logz_s = np.array([])
logzerr_s = np.array([])


"""-------------------------------------------------------------------------------------------
filename format : /home/tommy/lisaenv/blip/[mode]_n[nside]_s[seed#]/logz_rs[seed#]_[mode].txt
for nside=4, seed=10, mode=s: /home/tommy/lisaenv/blip/s_n4_s10/logz_rs10_s.txt
----------------------------------------------------------------------------------------------"""
for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/" + mode1 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode1 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    # logz_s = np.append(logz_s, float("{:.8e}".format(float(last_line))))
    logz_s = np.append(logz_s, float(last_line))


"""----------------------------------------------------------------------------------------------
filename format : /home/tommy/lisaenv/blip/[mode]_n[nside]_s[seed#]/logzerr_rs[seed#]_[mode].txt
for nside=4, seed=10, mode=s: /home/tommy/lisaenv/blip/s_n4_s10/logzerr_rs10_s.txt
-------------------------------------------------------------------------------------------------"""
for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/" + mode1 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logzerr_rs" + str(seeds[x]) + "_" + mode1 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    # logzerr_s = np.append(logzerr_s, float("{:.3e}".format(float(last_line))))
    logzerr_s = np.append(logzerr_s, float(last_line))


"""----------
Orbiting case
-------------"""
mode2 = 'o'

logz_o = np.array([])
logzerr_o = np.array([])

for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/" + mode2 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logz_rs" + str(seeds[x]) + "_" + mode2 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    # logz_o = np.append(logz_o, float("{:.3e}".format(float(last_line))))
    logz_o = np.append(logz_o, float(last_line))


for x in range(len(seeds)):
    file = open("/home/tommy/lisaenv/blip/" + mode2 + "_n" + str(nside) + "_s" + str(seeds[x]) + "/logzerr_rs" + str(seeds[x]) + "_" + mode2 + ".txt")
    last_line = file.readlines()[-1].replace("\n",'')
    # logzerr_o = np.append(logzerr_o, float("{:.3e}".format(float(last_line))))
    logzerr_o = np.append(logzerr_o, float(last_line))

logz_o = np.array(logz_o, dtype=np.float128)
logz_s = np.array(logz_s, dtype=np.float128)
logzerr_o = np.array(logzerr_o, dtype=np.float128)
logzerr_s = np.array(logzerr_s, dtype=np.float128)

print(10**(abs(logz_o - logz_s)))
"""-----
ANALYSIS
--------"""






# z_o = np.array(10**logz_o, dtype=np.float128)
# inv_z_s = np.array(10**-logz_s, dtype=np.float128)
# print(inv_z_s)
# zerr_o = 10**logzerr_o
# zerr_s = 10**logzerr_s
# print(zerr_o, zerr_s)
# import pdb; pdb.set_trace()
# print(logz_o - logz_s)
# bayes_factor = 10**(logz_o - logz_s)
# print("BayesFactor =", bayes_factor)

# bf_err = np.sqrt(zerr_o**2 + (zerr_s * bayes_factor)**2)
# # bf_err = np.array(np.sqrt(zerr_o**2 * z_s**-2 + zerr_s**2 * (z_o/z_s**2)**2), dtype=np.float128)
# print(bf_err)

# plt.errorbar(seeds, bayes_factor, yerr=bf_err, fmt='o')
# plt.title("Bayes' Factor of Orbiting vs Stationary for seeds 10,20,30,40,50,60")
# plt.xlabel('Seeds #')
# plt.ylabel("Bayes' Factor")
# plt.show()