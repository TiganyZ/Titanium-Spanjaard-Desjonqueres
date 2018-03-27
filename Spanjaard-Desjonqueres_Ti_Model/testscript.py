#! /usr/bin/env python
import subprocess, math, time, sys
from optparse import OptionParser

def write_and_exit (ObjectiveFunction, filename):
    retval = subprocess.call('rm -f log.d', shell=True)
    fmin_file = open(filename,mode='a')
    fmin_file.write(str(ObjectiveFunction)+'\n')
    fmin_file.close()
    sys.exit()

filename = 'fmin.val'
ext = 'spanj'
vargs = ' '
BIN = '~/lm/build/'
# get filename, ctrl extension, in-line parameters
filename = 'fmin.val'
cargs = ''
ctrlf = 'ctrl.spanj'



######## Get Fermi energy of system 

def fermi_energy(ctrlfile, BINpath, args):
	file = open('out',mode='w')
	cmd = BINpath + 'tbe --mxq' + ' ' + args + ' ' + ctrlfile
	retval = subprocess.call(cmd, shell=True, stdout=file)
	file.close()
	file = open('out',mode='r')
	cmd = "grep 'Fermi energy:' out | tail -1 | awk '{print $4}' | sed 's/.$//'"
	proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	out,err = proc.communicate()
	E_F = out
	print(out) 
	#etot_bcc = float(out[0:-1])
	file.close()
	return E_F[:-2]

######## Find width of D bands 
def symfile_def(ext, nit):
	symfile = open('syml.' + ext,mode='w')
	symfile.write(nit + '  	0 0 0  0 0 1 G to H\n')
	symfile.write(nit + '   0 0 1  0.5 0.5 0.5 H to P\n')
	symfile.write(nit + '   0.5 0.5 0.5  0 0.5 0.5 P to N\n')
	symfile.write(nit + '   0 0.5 0.5  0 0 0 N to G\n')
	symfile.close()
	
def dband_width(ctrlfile, E_F, BINpath, args, symfil):
	file = open('out',mode='w')
	#cmd = BIN + 'tbe -vbcc=1 --mxq --band ' +' ' + ext #cmd = BIN + 'tbe -vbcc=1' + ctrlfile +' -ef=' + str(E_Fermi) + ' --band~fn=syml '# + ' ' + ext
	cmd=BINpath + 'tbe ' + ctrlfile + ' -ef=' + E_F + ' ' + args + ' ' + '--band~fn=' + symfil
	print('band command', cmd)
	retval = subprocess.call(cmd, shell=True, stdout=file)
	file.close()
	file = open('out',mode='r') 
	cmd = "grep 'total energy' out | tail -1 | awk '{print $4}'"
	proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	out,err = proc.communicate()
	print('etot=',out)
	etot_bcc = float(out[0:-1])
	print (etot_bcc)
	file.close()
	bndfile = open('bnds.' + ext,mode='r')
	lines = [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
	j = 0
	while j < 10:
		lines[j] = bndfile.readline()[0:-1]
		print(lines[j])
		j += 1
	d_minimum = ( float(lines[6].split()[0]) + float(lines[7].split()[0]) ) / 2.0
	d_maximum = ( float(lines[6].split()[4]) + float(lines[7].split()[4]) ) / 2.0
	d_width = (d_maximum - d_minimum)*13.606
	d_width2= -float(lines[3].split()[0]) + float(lines[4].split()[4])
	print ('d_width', d_width, d_width2*13.606)



def plot_bands(ylim):
	cmd = 'echo ' + ylim + ' / | ~/lm/build/plbnds -fplot -ef=0 -scl=13.606 -lbl=G,H,P,N,G bnds.spanj'
	retval = subprocess.call(cmd, shell=True)
	cmd = '~/lm/build/fplot -f plot.plbnds'
	retval = subprocess.call(cmd, shell=True)
	cmd = 'cp fplot.ps tispanj.ps'
	retval = subprocess.call(cmd, shell=True)
	
	
argso = ' '#'-vspanjddd='

#E_Fermi = fermi_energy(ctrlf, BIN, args)
#dband_width(ctrlf, E_Fermi, BIN, args)
#plot_bands()
nit=str(11)
#symfile_def('spanj', nit)
symfile='syml'
numddd= 0.0045715
for i in range(1):
	args = argso + str((numddd + i*0.001))
	E_Fermi = fermi_energy(ctrlf, BIN, args)
	dband_width(ctrlf, E_Fermi, BIN, args, symfile)
plot_bands('-22,12')
	

######## Produce Bands of Rutile

#cmd = 'tbe ' + ctrlfile + ' -ef=' + str(E_Fermi) + '--mxq --band~fn'
#echo -22,12 / | ~/lm/build/plbnds -fplot -ef=0 -scl=13.606 -lbl=G,X,R,Z,G,M,A,Z bnds.rutile2
#~/lm/build/fplot -f plot.plbnds
#cp fplot.ps rutilebandslocal.ps
