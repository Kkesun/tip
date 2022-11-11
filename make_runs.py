#!/usr/bin/env python
#This script will make the data files required for the ternary potential and frozen fluid method
import numpy as np
import sys,os
import subprocess
from pylab import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------------------------
# User inputs
#----------------------------------------------------------------------------------------------------------------------------------

savetag='tip_10_25_angle_30_CA_150_Liquid_at_30_test' # This is an identifier that will be used in each saved lowest filename, and the data_log entries. This should be changed each run

# Surface properties
NPOSTX  = 0 #Number of posts in x direction
NPOSTY  = 0 #Number of posts in y direction
WIDTHX  = 0 #Width of pillar in x
WIDTHY  = 0 #Width of pillar in y
TOPEXX  = 0 #Extension of cap out from pillar in x
TOPEXY  = 0 #Extension of cap out from pillar in y
HEIGHT1 = 0 #Pillar height
HEIGHT2 = 0 #Thickness of reentrant cap
LIPX    = 0 #x-thickness of doubly reentrant lip
LIPY    = 0 #y-thickness of doubly reentrant lip
LIPZ    = 0 #lip depth
GRIDX = 80 #Number of nodes in x
GRIDY = 80  #Number of nodes in y
GRIDZ = 100 #Number of nodes in z

#Frozen fluid method parameters
CONF_STRENGTH=-1.0 # Magnitude of the confining potential
LAMBDA=0 # Boyer stabilisation parameter (use 50.0 for highly wetting liquids (CA<30), otherwise set to 0)

N_PHASE = 3 #Number of phases in the system
alpha=1.0 #Interface width [L.U.]

gamma_12=1.0-np.cos(np.radians(150.0)) #gamma_12 interfacial tension
gamma_13=1.0 #gamma_13 interfacial tension
gamma_23=1.0 #gamma_23 interfacial tension

const_label_1=2 #Constrain of phase 1: 0=no constraint, 1=volume constraint, 2=pressure constraint
const_value_1=-0.001 #Value we wish to constrain phase 1 to
const_strength_1=0.0001 #Magnitude of contraint strength (only used for volume constraints)

const_label_2=1 #Constrain of phase 2: 0=no constraint, 1=volume constraint, 2=pressure constraint
const_value_2=GRIDX*GRIDY*30.0 #Value we wish to constrain phase 2 to
const_strength_2=0.001 #Magnitude of contraint strength (only used for volume constraints)

const_label_3=0 #Constrain of phase 3: 0=no constraint, 1=volume constraint, 2=pressure constraint
const_value_3=0.0 #Value we wish to constrain phase 3 to
const_strength_3=0.0 #Magnitude of contraint strength (only used for volume constraints)

PS_1=1 #0=fix this phase so it is not changed during minimisation, 1=vary this phase
PS_2=1 #0=fix this phase so it is not changed during minimisation, 1=vary this phase
PS_3=1 #0=fix this phase so it is not changed during minimisation, 1=vary this phase

#Bottom surface wetting properties
gamma_1s=0.0 #gamma_1surface
gamma_2s=0.0 #gamma_2surface
gamma_3s=0.0 #gamma_2surface


#Mininmisation algorithm parameters
PATHTOGMIN="../../GMIN" # Write the path to the script GMIN from the directory where this script is run from
INIT_COORDS0='.TRUE.'
MAX_ITERATIONS_A=30 # Number of iterations to relax the diffuse solid
MAX_ITERATIONS_B=100000 # Maximum number of iterations to minimise the energy of the full system

dirname='.' #Name of directory to make the data folder in

#Looping array for height of tip
#data_array=np.linspace(30,10,21)
#data_array=np.concatenate([data_array,np.linspace(11,40,30)])
#data_array = np.concatenate([np.arange(20,50,0.5),np.arange(50,20,0.5)])
data_array=[25]
#data_array = np.flip(np.arange(20,40,1.0))
#data_array = np.arange(20,40,1.0)
#data_array = np.append(data_ext,data_retr)

#----------------------------------------------------------------------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------------------------------------------------------------------

# Make the input parameters
def make_data():
	#Compute the kappas and kappa_primes
	k_1=3.0/alpha*(gamma_12+gamma_13-gamma_23)
	k_2=3.0/alpha*(gamma_12-gamma_13+gamma_23)
	k_3=3.0/alpha*(-gamma_12+gamma_13+gamma_23)

	#Define the kappa'
	kp_1=k_1
	kp_2=k_2
	kp_3=k_3

	A=0
	#Amin=2*k_2
	#if Amin <= 0:
	#	#The system is in a spreading setup, switch on the spreading term A
	#	Amin=-Amin
	#	Amax=2*np.min([k_1,k_3])
	#	#print('a',Amin,Amax)
	#	A=Amin+0.9*(Amax-Amin)
	#		
	#	k_1=k_1-0.5*A
	#	k_2=k_2+0.5*A
	#	k_3=k_3-0.5*A
	#	
	#	kp_1=alpha**2*(k_1+A/2.0)
	#	kp_2=alpha**2*(k_2-A/2.0)
	#	kp_3=alpha**2*(k_3+A/2.0)
	#else:
	#	A=0

	#Define the wetting parameters of the bottom of the system
	g_1=12*gamma_1s
	g_2=12*gamma_2s
	g_3=12*gamma_3s

	#Now shift the wetting parameters to minimise interference of wetting gradients with diffuse interface
	min_g=np.min([g_1,g_2,g_3])
	g_1=g_1-min_g
	g_2=g_2-min_g
	g_3=g_3-min_g

	
	#Write the data files
	#Check if 'data' folder exists
	if not os.path.exists(dirname+'/data'):
		os.mkdir(dirname+'/data')
		
	#Write kappa
	with open(dirname+'/data'+'/K.in','w') as f:
		f.write("%s\n"%k_1)
		f.write("%s\n"%k_2)
		f.write("%s\n"%k_3)
		f.write("%s\n"%A)
		
	#Write kappa'
	with open(dirname+'/data'+'/KP.in','w') as f:
		f.write("%s\n"%kp_1)
		f.write("%s\n"%kp_2)
		f.write("%s\n"%kp_3)

	#Write wetting parameters
	with open(dirname+'/data'+'/wetenergy.in','w') as f:
		f.write("%s\n"%g_1)
		f.write("%s\n"%g_2)
		f.write("%s\n"%g_3)
		
	#Write constraints
	with open(dirname+'/data'+'/constraints.in','w') as f:
		f.write("%s %s %s\n"%(const_label_1,const_value_1,const_strength_1))
		f.write("%s %s %s\n"%(const_label_2,const_value_2,const_strength_2))
		f.write("%s %s %s\n"%(const_label_3,const_value_3,const_strength_3))
		
	#Write phase switches
	with open(dirname+'/data'+'/phaseswitch.in','w') as f:
		f.write("%s\n"%PS_1)
		f.write("%s\n"%PS_2)
		f.write("%s\n"%PS_3)
	return


# Write the data.in file
def write_data():
	with open("data.in",'w') as f:
		f.write("%s\n"%NPOSTX)
		f.write("%s\n"%NPOSTY)
		f.write("%s\n"%WIDTHX)
		f.write("%s\n"%WIDTHY)
		f.write("%s\n"%TOPEXX)
		f.write("%s\n"%TOPEXY)
		f.write("%s\n"%HEIGHT1)
		f.write("%s\n"%HEIGHT2)
		f.write("%s\n"%LIPX)
		f.write("%s\n"%LIPY)
		f.write("%s\n"%LIPZ)
		f.write("%s\n"%GRIDX)
		f.write("%s\n"%GRIDY)
		f.write("%s\n"%GRIDZ)

		f.write("%s\n"%N_PHASE)
		f.write("%s\n"%LAMBDA)
		f.write("%s\n"%CONF_STRENGTH)
		
		f.write("%s\n"%INIT_COORDS)
		f.write("%s\n"%MAX_ITERATIONS)
		
		f.write("%s\n"%ZCOM)
	return
	

		
# Make the inital coords
def make_coords(startstat):

	if (startstat):
		coords_4d=np.zeros([GRIDX,GRIDY,GRIDZ,(N_PHASE-1)],dtype=float)
			
		coords=np.zeros([GRIDX*GRIDY*GRIDZ*(N_PHASE-1)],dtype=float)
		for j1 in range(0,GRIDX):
			for j2 in range(0,GRIDY):
				for j3 in range(0,GRIDZ):
					for j4 in range(0,N_PHASE-1):
						cur=j1*GRIDY*GRIDZ*(N_PHASE-1)+j2*GRIDZ*(N_PHASE-1)+j3*(N_PHASE-1)+j4
						coords[cur]=coords_4d[j1,j2,j3,j4]	
	else:
		fid=open("lowests",'r')
		coords=np.loadtxt(fid,skiprows=0)
		fid.close
		
		if firstiterstat==True:
			for j1 in range(0,GRIDX):
				for j2 in range(0,GRIDY):
					for j3 in range(0,GRIDZ):
						if j3<=20:				
							cur=j1*GRIDY*GRIDZ+j2*GRIDZ+j3
							coords[2*cur+1]=1-coords[2*cur]
		else:						
			fid=open("lowests_saved",'r')
			coords_saved=np.loadtxt(fid,skiprows=0)
			fid.close		
					
			for j1 in range(0,GRIDX):
				for j2 in range(0,GRIDY):
					for j3 in range(0,GRIDZ):
						#if np.sqrt((j1-GRIDX/2.0)**2+(j2-GRIDY/2.0+20)**2+(j3-GRIDZ/2.0)**2) <=20:					
						cur=j1*GRIDY*GRIDZ+j2*GRIDZ+j3
						if coords[2*cur]<=0.5:
							coords[2*cur+1]=coords_saved[2*cur+1]-coords[2*cur]	
		

	#Write coords to file					
	with open("coords",'w') as f:
		s1=0
		for j1 in range(0,GRIDX):
			for j2 in range(0,GRIDY):
				for j3 in range(0,GRIDZ):
					for j4 in range(0,N_PHASE-1):				
						f.write("%s\n"%coords[s1])
						s1=s1+1

	debug=False
	if debug==True:
		#Format lowests for analysis
		lowests=np.reshape(coords,[GRIDX,GRIDY,GRIDZ,N_PHASE-1],'C')
		#lowests=np.rot90(lowests, k=-1, axes=(1,2))
	
		#Compute the 1-2 contour
		rc('axes', linewidth=2)
		fig, axs = plt.subplots(3)
		axs[0].contourf(lowests[0,:,:,0])
		axs[0].set_aspect('equal')
		axs[1].contourf(lowests[0,:,:,1])
		axs[1].set_aspect('equal')
		axs[2].contourf(1-lowests[0,:,:,0]-lowests[0,:,:,1])
		axs[2].set_aspect('equal')
		
		plt.show()
		quit()

	return 

#Extract the energy of the current system
def get_energy():
	with open('lowest','r') as f:
		lines=f.readlines()
		line=lines[2]
		words=line.split()
		energy=words[2]
	return energy
#----------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------
# Compile minimisation code
#----------------------------------------------------------------------------------------------------------------------------------
subprocess.run([PATHTOGMIN,"-n"])
#----------------------------------------------------------------------------------------------------------------------------------
	

	
#----------------------------------------------------------------------------------------------------------------------------------
# Main
#----------------------------------------------------------------------------------------------------------------------------------
lg=len(data_array)

f_data=open('data_log','a')
f_data.write(("%s "*8 + "\n") %("savetag","ig","ZCOM","gamma_12","gamma_13","gamma_23","const_value_2","energy"))
f_data.close()

firstiterstat=True
for ig in range(0,lg):

	print('Step ', ig, 'data= ', data_array[ig])

	ZCOM=data_array[ig]
	with open('zcom.in','w') as f:
		f.write(("%s" + "\n") %(ZCOM))

	#1) Make the solid system--------------------------------------------------------------------------------------
	#Write the data files
	INIT_COORDS=INIT_COORDS0
	PS_1=1
	PS_2=0
	const_label_2_saved=const_label_2
	const_label_2=0
	MAX_ITERATIONS=MAX_ITERATIONS_A
	make_data()
	write_data()
	make_coords(True)

	#Run the minimisation
	subprocess.run(["./gmin"])
	energy=get_energy()
	energy=float(energy)

	#Save the minimised state
	lowestname='lowest_'+savetag+'_'+str(ig)+'_'+'INIT'	
	subprocess.run(["cp","lowest",str(lowestname)])

	#Copy the minimised state to initialise the next iteration
	fid=open("lowest",'r')
	lowests=np.loadtxt(fid,skiprows=4)
	fid.close
	np.savetxt("lowests",lowests)
	subprocess.run(["cp","lowests","coords"])	


	#2) Perform the minimisation-----------------------------------------------------------------------------------
	#Write the data files
	INIT_COORDS='.FALSE.'
	PS_1=0
	PS_2=1
	const_label_2=const_label_2_saved
	MAX_ITERATIONS=MAX_ITERATIONS_B	
	make_data()
	write_data()
	make_coords(False)

	#Run the minimisation
	subprocess.run(["./gmin"])
	energy=get_energy()
	energy=float(energy)

	#Save the minimised state
	lowestname='lowest_'+savetag+'_'+str(ig)	
	subprocess.run(["cp","lowest",str(lowestname)])

	#Copy the minimised state to initialise the next iteration
	fid=open("lowest",'r')
	lowests=np.loadtxt(fid,skiprows=4)
	fid.close
	np.savetxt("lowests",lowests)
	subprocess.run(["cp","lowests","lowests_saved"])


	f_data=open('data_log','a')
	f_data.write(("%s "*8 + "\n") %(savetag,ig,ZCOM,gamma_12,gamma_13,gamma_23,const_value_2,energy))
	f_data.close()
						
	firstiterstat=False
#----------------------------------------------------------------------------------------------------------------------------------	


