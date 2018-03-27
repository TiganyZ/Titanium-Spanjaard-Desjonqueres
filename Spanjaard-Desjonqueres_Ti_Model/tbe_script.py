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
ext = 'spd'
vargs = ' '
BIN = '/lm/build'
# get filename, ctrl extension, in-line parameters
filename = 'fmin.val'
cargs = ''
ctrl_args = False
cmd_pars = False
parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="filename", help="fmin.val alternative filename")
parser.add_option("-e", "--ext",  action="store", type="string", dest="ext", help="ctrl file extension name")
parser.add_option("-p", "--parameters",  action="store", type="string", dest="par", help="starting paramters")
parser.add_option("-a", "--arguments",  action="store", type="string", dest="cargs", help="ctrl file arguments")
parser.add_option("-b", "--bindir",  action="store", type="string", dest="BIN", help="directory for binary sauce")
(options, args) = parser.parse_args()
if options.filename != None:
    filename = options.filename
if options.ext != None:
    ext = options.ext
if options.cargs != None:
    cargs = options.cargs
    ctrl_args = True
if options.par != None:
    par = options.par
    cmd_pars = True
if options.BIN != None:
    BIN = options.BIN

# get args from file or command line
if cmd_pars:
    vals = par.split()
else:
    fmin_file = open(filename,mode='r')
    for line in fmin_file:
        vals = line.split()
    par = line
    fmin_file.close()
vargs = ''
for val in vals:
    vargs = vargs + " -v" + val
vargs = vargs +  " "
vargs = " " + cargs + " " + vargs

if ctrl_args:
    print sys.argv[0],'starting.. ext =',ext,', file =',filename,' ctrl_args = ', cargs,', vals =',par, 'binaries in', BIN
else:
    print sys.argv[0],'starting.. ext =',ext,', file =',filename,', vals =',par, 'binaries in', BIN

BIN = BIN + '/'

OBJ = [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
WGT = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# bcc lattice constant
WGT[0] = 50
# hcp properties
WGT[6] = 0.2
WGT[7] = 0.2
WGT[8] = 0.5

# get K, a
K_bcc_exp, a_bcc_exp, mmom_bcc_target, K_fcc_exp, a_fcc_exp, a_hcp_exp, K_hcp_exp, e_hcpbcc_gga = 168.0, 5.4235, 2.6, 133.0, 6.8332, 4.74298, 160.0, 14.8

# bcc ..
retval = subprocess.call('rm -f save.' + ext, shell=True)
vf = 0.8
while vf < 1.4:
    arg = '-vvfrac=' + str(vf) + ' '
    file = open('out',mode='w')
    cmd = BIN + 'tbe -vbcc=1 --mxq ' + arg + ' ' + vargs + ext
    retval = subprocess.call(cmd, shell=True, stdout=file)
    file.close()
    vf += 0.05
file = open('pt-spd-bcc',mode='w')
retval = subprocess.call(BIN + 'vextract c vfrac etot < save.' + ext, shell=True, stdout=file)
file.close()
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 5 pt-spd-bcc | grep 'gradzr returned' | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    ierr = int(out[0:-1])
except:
    print sys.argv[0],'cannot read trial bcc output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
if ierr != 0:
    print sys.argv[0],'no minimum in bcc E-V curve, gradzr returned', ierr,' leaving ...'
    ObjectiveFunction=1000000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 5 pt-spd-bcc | grep min= | awk '{print $7}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    VF = out[0:-1]
except:
    print sys.argv[0],'cannot read bcc output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
if VF == '':
    print sys.argv[0],'emtpy string; leaving ...'
    ObjectiveFunction=1000000
    write_and_exit (ObjectiveFunction, filename)
vf = float(VF)
if vf < 0.5 or vf > 1.5:
    print sys.argv[0],'vfrac =',vf,', out of range; leaving ...'
    ObjectiveFunction=500000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo " + VF + " | " + BIN + "pfit -nc=2 5 pt-spd-bcc | grep f= | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
Epp = float(out[0:-1])
V0 = 79.76508
K_bcc = (vf / V0) * Epp * 14700
V_bcc = vf * V0 * 2
a_bcc = math.pow( V_bcc , 1.0/3.0 )
diff = (a_bcc - a_bcc_exp)
OBJ[0] = math.pow( diff , 2 )
diff = (K_bcc - K_bcc_exp)
OBJ[1] = math.pow( diff , 2 )
# do an equilibrium calculation
# symmetry line file for Gamma  and H (d-band width) points
symfile = open('syml.' + ext,mode='w')
symfile.write('2  0 0 0    0 0 1\n')
symfile.write('0    0 0 0  0 0 0\n')
symfile.close()
arg = '-vvfrac=' + str(vf) + ' '
file = open('out',mode='w')
cmd = BIN + 'tbe -vbcc=1 --mxq --band ' + arg + ' ' + vargs + ext
retval = subprocess.call(cmd, shell=True, stdout=file)
file.close()
file = open('out',mode='r')
cmd = "grep 'Stoner potentials J=' out | tail -1 | awk '{print $7}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
mmom_bcc = float(out[0:-1]) 
cmd = "grep 'total energy' out | tail -1 | awk '{print $4}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
etot_bcc = float(out[0:-1])
file.close()
bndfile = open('bnds.' + ext,mode='r')
lines = [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
j = 0
while j < 10:
    lines[j] = bndfile.readline()[0:-1]
    j += 1
d_minimum = ( float(lines[7].split()[0]) + float(lines[9].split()[0]) ) / 2.0
d_maximum = ( float(lines[7].split()[4]) + float(lines[9].split()[4]) ) / 2.0
d_width = d_maximum - d_minimum
d_width_lda = 0.4
diff = (d_width - d_width_lda)
OBJ[2] = math.pow( diff , 2 )
diff = (mmom_bcc - mmom_bcc_target)
OBJ[3] = math.pow( diff , 2 )
print 'Got bcc. V = %.2f a = %.3f, K = %.0f, etot = %.3f, moment = %.2f, W_d = %.2f '% (V_bcc/2, a_bcc, K_bcc, etot_bcc, mmom_bcc, d_width)
print '         Targets:  a = %.3f, K = %.0f, moment = %.1f. OBJECTIVE: %.2g (a) %.1f (K), %.2g (m),  weight  %.1f (a) %.1f (K) %.1f (m)' % (a_bcc_exp, K_bcc_exp, mmom_bcc_target, OBJ[0], OBJ[1], OBJ[3], WGT[0], WGT[1], WGT[3])
print '                   W_d = %.2f. OBJECTIVE: %.2f, weight  %.1f' % (d_width_lda, OBJ[2], WGT[2])

# get C'
Cp_exp = 53
V = V_bcc/2
retval = subprocess.call('rm -f save.' + ext, shell=True)
alpha = -0.03
while alpha < 0.04:
    arg = '-valpha=' + str(alpha) + ' -vvfrac=' + str(vf) + ' '
    file = open('out',mode='w')
    cmd = BIN + 'tbe -vbcc=1 -vnk=16 --mxq -vcp=T ' + arg + ' ' + vargs + ext
    retval = subprocess.call(cmd, shell=True, stdout=file)
    file.close()
    alpha = alpha + 0.01
file = open('pt-spd',mode='w')
retval = subprocess.call(BIN + 'vextract c alpha etot < save.' + ext, shell=True, stdout=file)
file.close()
cmd = "echo 0 | " + BIN + "pfit -nc=2 3 pt-spd | grep f= | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
Epp = float(out[0:-1])
Cp = (1.0/3.0) * (1/V) * Epp * 14700
diff = (Cp - Cp_exp)
OBJ[4] = math.pow( diff , 2 )
print "         C' = %f. Target: %.2f. OBJECTIVE: %.2f, weight  %.1f" % (Cp, Cp_exp, OBJ[4], WGT[4])

# get c_44
c_44_exp = 122
V = V_bcc/2
retval = subprocess.call('rm -f save.' + ext, shell=True)
alpha = -0.03
while alpha < 0.04:
    arg = '-valpha=' + str(alpha) + ' -vvfrac=' + str(vf) + ' '
    file = open('out',mode='w')
    cmd = BIN + 'tbe -vbcc=1 -vnk=16 --mxq -vc44=T ' + arg + ' ' + vargs + ext
    retval = subprocess.call(cmd, shell=True, stdout=file)
    file.close()
    alpha = alpha + 0.01
file = open('pt-spd',mode='w')
retval = subprocess.call(BIN + 'vextract c alpha etot < save.' + ext, shell=True, stdout=file)
file.close()
cmd = "echo 0 | " + BIN + "pfit -nc=2 3 pt-spd | grep f= | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
Epp = float(out[0:-1])
c_44 = (1.0/3.0) * (1/V) * Epp * 14700
diff = (c_44 - c_44_exp)
OBJ[5] = math.pow( diff , 2 )
print "         c_44 = %f. Target: %.2f. OBJECTIVE: %.2f, weight  %.1f" % (c_44, c_44_exp, OBJ[5], WGT[5])

# hcp ..
retval = subprocess.call('rm -f save.' + ext, shell=True)
vf = 0.8
while vf < 1.4:
    arg = '-vvfrac=' + str(vf) + ' '
    file = open('out',mode='w')
    cmd = BIN + 'tbe -vbcc=0 -vhcp=1 --mxq ' + arg + ' ' + vargs + ext
    retval = subprocess.call(cmd, shell=True, stdout=file)
    file.close()
    vf += 0.05
file = open('pt-spd-hcp',mode='w')
retval = subprocess.call(BIN + 'vextract c vfrac etot < save.' + ext, shell=True, stdout=file)
file.close()
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 5 pt-spd-hcp | grep 'gradzr returned' | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    ierr = int(out[0:-1])
except:
    print sys.argv[0],'cannot read trial hcp output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
if ierr != 0:
    print sys.argv[0],'no minimum in hcp E-V curve, gradzr returned', ierr,' leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 4 pt-spd-hcp | grep min= | awk '{print $7}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    VF = out[0:-1]
except:
    print sys.argv[0],'cannot read hcp output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
vf = float(VF)
if VF == '':
    print sys.argv[0],'emtpy string; leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
vf = float(VF)
if vf < 0.5 or vf > 1.5:
    print sys.argv[0],'vfrac =',vf,', out of range; leaving ...'
    ObjectiveFunction=50000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo " + VF + " | " + BIN + "pfit -nc=2 5 pt-spd-hcp | grep f= | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
Epp = float(out[0:-1])
V0 = 79.76508 * 2
K_hcp = (vf / V0) * Epp * 14700
V_hcp = vf * V0
a_hcp = math.pow( (2.0 * V_hcp / math.pow( 8 , 0.5 )) , 1.0/3.0 )
diff = (a_hcp - a_hcp_exp)
OBJ[6] = math.pow( diff , 2 )
diff = (K_hcp - K_hcp_exp)
OBJ[7] = math.pow( diff , 2 )
# do an equilibrium calculation
arg = '-vvfrac=' + str(vf) + ' '
file = open('out-hcp',mode='w')
cmd = BIN + 'tbe -vhcp=1 -vbcc=0 --mxq ' + arg + ' ' + vargs + ext
retval = subprocess.call(cmd, shell=True, stdout=file)
file.close()
file = open('out-hcp',mode='r')
cmd = "grep 'Stoner potentials J=' out-hcp | tail -1 | awk '{print $7}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
mmom_hcp = float(out[0:-1]) 
cmd = "grep 'total energy' out-hcp | tail -1 | awk '{print $4}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
etot_hcp = float(out[0:-1])/2
file.close()


# non magnetic hcp ..
retval = subprocess.call('rm -f save.' + ext, shell=True)
vf = 0.8
while vf < 1.4:
    arg = '-vvfrac=' + str(vf) + ' '
    file = open('out',mode='w')
    cmd = BIN + 'tbe -vbcc=0 -vhcp=1 -vnsp=1 -vul=0 ' + arg + ' ' + vargs + ext
    retval = subprocess.call(cmd, shell=True, stdout=file)
    file.close()
    vf += 0.05
file = open('pt-spd-hcp-NM',mode='w')
retval = subprocess.call(BIN + 'vextract t vfrac etot < save.' + ext, shell=True, stdout=file)
file.close()
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 5 pt-spd-hcp-NM | grep 'gradzr returned' | awk '{print $5}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    ierr = int(out[0:-1])
except:
    print sys.argv[0],'cannot read trial hcp output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
if ierr != 0:
    print sys.argv[0],'no minimum in hcp E-V curve, gradzr returned', ierr,' leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo '1\nmin' | " + BIN + "pfit -nc=2 4 pt-spd-hcp-NM | grep min= | awk '{print $7}'"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
try:
    VF = out[0:-1]
except:
    print sys.argv[0],'cannot read hcp output from pfit, leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
vf = float(VF)
if VF == '':
    print sys.argv[0],'emtpy string; leaving ...'
    ObjectiveFunction=100000
    write_and_exit (ObjectiveFunction, filename)
vf = float(VF)
if vf < 0.5 or vf > 1.5:
    print sys.argv[0],'vfrac =',vf,', out of range; leaving ...'
    ObjectiveFunction=50000
    write_and_exit (ObjectiveFunction, filename)
cmd = "echo " + VF + " | " + BIN + "pfit -nc=2 5 pt-spd-hcp-NM | grep f="
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
out,err = proc.communicate()
etot_hcp_nonmag = float(out.split()[1][2:])/2
Epp = float(out.split()[4])
V0 = 79.76508 * 2
K_hcp_nonmag = (vf / V0) * Epp * 14700
V_hcp_nonmag = vf * V0
a_hcp_nonmag = math.pow( (2.0 * V_hcp / math.pow( 8 , 0.5 )) , 1.0/3.0 )

print 'Got hcp. V = %.2f a = %.3f, K = %.0f, etot = %.3f, moment = %.2f '% (V_hcp/2, a_hcp, K_hcp, etot_hcp, mmom_hcp)
print '         Targets:  a = %.3f, K = %.0f. OBJECTIVE: %.2g (a) %.1f (K), weight  %.1f (a) %.1f (K)' % (a_hcp_exp, K_hcp_exp,OBJ[6], OBJ[7], WGT[6], WGT[7])
print '         Non magnetic: V = %.2f a = %.3f, K = %.0f, emag = %.4f' % (V_hcp_nonmag/2, a_hcp_nonmag, K_hcp_nonmag, etot_hcp_nonmag-etot_hcp)
e_hcpbcc = 1000 * ( etot_hcp - etot_bcc )
diff = e_hcpbcc - e_hcpbcc_gga
OBJ[8] = math.pow( diff , 2 )
print '         e_hcp - e_bcc = %2.fmRy, Target = %2.fmRy. OBJECTIVE: %f , weight  %.1f' % (e_hcpbcc, e_hcpbcc_gga, OBJ[8], WGT[8])

# Make objective function
ObjectiveFunction = 0.0
i = 0
while i < 9:
    ObjectiveFunction += OBJ[i] * WGT[i]
    i += 1
print 'Objective function: %f' % (ObjectiveFunction)

write_and_exit (ObjectiveFunction, filename)
