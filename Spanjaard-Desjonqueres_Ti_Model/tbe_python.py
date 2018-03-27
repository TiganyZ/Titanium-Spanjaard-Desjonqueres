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
`etot_bcc = float(out[0:-1])
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
