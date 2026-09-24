import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('/glade/u/home/cobrien/plots/paper.mplstyle')
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Kaipy modules
import kaipy.kaiViz as kv
import kaipy.gamera.msphViz as mviz
import kaipy.remix.remix as remix
import kaipy.gamera.magsphere as msph
import kaipy.gamera.gampp as gampp
import kaipy.gamera.rcmpp as rcmpp

def create_command_line_parser():
    """Create the command-line argument parser.

    Create the parser for command-line arguments, lifted from gamsphVid

    Returns:
        argparse.ArgumentParser: Command-line argument parser for this script.
    """
    #Defaults
    fdirfile = "fdirs.json"
    oDir = "vid2D"
    ts = 0 # Start time [min]
    te = 1441 # End time [min]
    dt = 60.0 # Timesteps [sec]
    noIon = False
    noRCM = False
    Nblk = 1 #Number of blocks
    nID = 1 #Block ID of this job
    doJy = False
    doBz = False
    doBigRCM = False
    doRestart = False

    MainS = """Creates simple multi-panel figure for three MAGE runs
    Left Panel - Residual vertical magnetic field for run 1
    Middle Panel - Residual vertical magnetic field for run 2
    Right Panel - Residual vertical magnetic field for run 3
    Insets - Northern and Southern hemisphere FAC patterns for each run
    """

    parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-df',type=str,metavar="file",default=fdirfile,help="File containing directories and tags of runs to plot (default: %(default)s)")
    parser.add_argument('-o',type=str,metavar="directory",default=oDir,help="Subdirectory to write to (default: %(default)s)")
    parser.add_argument('-ts' ,type=int,metavar="tStart",default=ts,help="Starting time [min] (default: %(default)s)")
    parser.add_argument('-te' ,type=int,metavar="tEnd"  ,default=te,help="Ending time   [min] (default: %(default)s)")
    parser.add_argument('-dt' ,type=int,metavar="dt"    ,default=dt,help="Cadence       [sec] (default: %(default)s)")
    parser.add_argument('-Nblk' ,type=int,metavar="Nblk",default=Nblk,help="Number of job blocks (default: %(default)s)")
    parser.add_argument('-nID' ,type=int,metavar="nID"  ,default=nID,help="Block ID of this job [1-Nblk] (default: %(default)s)")
    #parser.add_argument('-nompi', action='store_true', default=noMPI,help="Don't show MPI boundaries (default: %(default)s)")
    parser.add_argument('-bz'   , action='store_true', default=doBz ,help="Show Bz instead of dBz (default: %(default)s)")
    parser.add_argument('-bigrcm', action='store_true',default=doBigRCM,help="Show entire RCM domain (default: %(default)s)")
    parser.add_argument('-noion', action='store_true', default=noIon,help="Don't show ReMIX data (default: %(default)s)")
    parser.add_argument('-norcm', action='store_true', default=noRCM,help="Don't show RCM data (default: %(default)s)")
    parser.add_argument('-restart', action='store_true', default=doRestart,help="Is this a restart of a prior script run (should the numbering not start at 0)? (default: %(default)s)")

    mviz.AddSizeArgs(parser)

    return parser

def main():
    #Defaults
    doDen = False
    doMPI = False #[Add MPI tiling]
    doBig = False #[Use big window]
    noMPI = False
    # Set up the command-line parser.
    parser = create_command_line_parser()
    #Finalize parsing
    args = parser.parse_args()
    
    # Load the json with the file directories and run tags 
    with open(args.df, 'r') as f:
        stage = json.load(f)
        fdirs = stage[0]
        ftags = stage[1]
    
    ts  = args.ts
    te  = args.te
    dt  = args.dt
    oDir = args.o
    Nblk = args.Nblk
    nID = args.nID
    doBz = args.bz
    doBigRCM = args.bigrcm
    doRestart = args.restart
    noIon = args.noion
    noRCM = args.norcm

    # Initialize the data
    gsphs = []
    rmcs = []
    remixdirs = []
    for i, fdir in enumerate(fdirs):
        gsphs.append(msph.GamsphPipe(fdir,ftags[i])) #Open the pipe, append to the pipe
        #Check for remix
        rcmChk = fdir + "/%s.mhdrcm.h5"%(ftags[i])
        rmxChk = fdir + "/%s.mix.h5"%(ftags[i])
        remixdirs.append(rmxChk)
        doRCM = os.path.exists(rcmChk)
        doMIX = os.path.exists(rmxChk)

        if (doRCM and (not noRCM)):
            print("Found RCM data")
            rcms.append(gampp.GameraPipe(fdir,ftags[i]+".mhdrcm"))
            mviz.vP = kv.genNorm(1.0e-2,100.0,doLog=True)
            rcmpp.doEll = not doBigRCM
        if (doMIX and (not noIon)):
            print("Found ReMIX data")

    #Setup timing info NOTE: Only relevant for making a video
    tOut = np.arange(ts*dt,te*dt,dt)
    Nt = len(tOut)
    vO = np.arange(ts, Nt + ts)
    i0 = 0 #NOTE: these indices are the step numbers and will change if there are multiple job blocks
    i1 = Nt

    #Setup figure
    figSz = (18,7.5) #TODO: make this calculated from the number of passed runs
    fig = plt.figure(figsize=figSz)
    gs = gridspec.GridSpec(3,9,height_ratios=[20,1,1],hspace=0.025)
    xyBds = mviz.GetSizeBds(args) # NOTE: This is a very silly funtion that sets the bounds to a few arbitrary sizes


    AxL = fig.add_subplot(gs[0,0:3]) # Left run axes
    AxM = fig.add_subplot(gs[0,3:6]) # Middle run axes
    AxR = fig.add_subplot(gs[0,6:]) # Right run axes

    AxC1 = fig.add_subplot(gs[-1,1:4]) # Residual field colorbar
    AxC2 = fig.add_subplot(gs[-1,5:8]) # FAC colorbar

    cbM = kv.genCB(AxC2,kv.genNorm(remix.facMax),"FAC",cM=remix.facCM,Ntk=4) # Make the colorbar norm for the FAC

    #Loop over sub-range
    for i in range(i0,i1):
        #Convert time (in seconds) to Step #
        nStp = np.abs(gsphs[0].T-tOut[i]).argmin()+gsphs[0].s0
        print("Minute = %5.2f / Step = %d"%(tOut[i]/dt,nStp))
        npl = vO[i]

        AxL.clear()
        AxM.clear()
        AxR.clear()

        BzL = mviz.PlotEqB(gsphs[0],nStp,xyBds,AxL,AxC1,doBz=doBz) # Left plot (run 1)
        AxL.set_title('High-Driving')
        BzM = mviz.PlotEqB(gsphs[1],nStp,xyBds,AxM,AxC1,doBz=doBz) # Middle plot (run 2)
        AxM.set_yticks([])
        AxM.set_ylabel('')
        AxM.set_title('Mean-Driving')
        BzR = mviz.PlotEqB(gsphs[2],nStp,xyBds,AxR,AxC1,doBz=doBz) # Right plot (run 3)
        AxR.yaxis.tick_right()
        AxR.yaxis.set_label_position('right')
        AxR.set_title('Low-Driving')

        gsphs[0].AddTime(nStp,AxL,xy=[0.825,0.89],fs="small") # Add the time to just the left plot
        gsphs[0].AddSW(nStp,AxL,xy=[0.700,0.025],fs="x-small") # Add the solar wind to every plot (it could be different)
        gsphs[1].AddSW(nStp,AxM,xy=[0.700,0.025],fs="x-small")
        gsphs[2].AddSW(nStp,AxR,xy=[0.700,0.025],fs="x-small")

        #Add inset RCM plot
        if (doRCM and (not noRCM)): #This is generally not what we're doing
            AxRCML = inset_axes(AxL,width="30%",height="30%",loc=3)
            rcmpp.RCMInset(AxRCML,rcms[0],nStp,mviz.vP)
            AxRCML.contour(kv.reWrap(gsphs[0].xxc),kv.reWrap(gsphs[0].yyc),kv.reWrap(Bz0),[0.0],colors=mviz.bz0Col,linewidths=mviz.cLW)
            rcmpp.AddRCMBox(AxL)

            AxRCMM = inset_axes(AxM,width="30%",height="30%",loc=3)
            rcmpp.RCMInset(AxRCMM,rcms[1],nStp,mviz.vP)
            AxRCMM.contour(kv.reWrap(gsphs[1].xxc),kv.reWrap(gsphs[1].yyc),kv.reWrap(Bz1),[0.0],colors=mviz.bz0Col,linewidths=mviz.cLW)
            rcmpp.AddRCMBox(AxM)

            AxRCMR = inset_axes(AxR,width="30%",height="30%",loc=3)
            rcmpp.RCMInset(AxRCMR,rcms[2],nStp,mviz.vP)
            AxRCMR.contour(kv.reWrap(gsphs[2].xxc),kv.reWrap(gsphs[2].yyc),kv.reWrap(Bz2),[0.0],colors=mviz.bz0Col,linewidths=mviz.cLW)
            rcmpp.AddRCMBox(AxR)

        if (doMIX and (not noIon)): #This is generally what we're doing
            print('Doing REMIX')
            ion0 = remix.remix(remixdirs[0],nStp)
            # gsphs[0].AddCPCP(nStp,AxL,xy=[0.610,0.925])
            mviz.AddIonBoxes(gs[0,0:3],ion0)

            ion1 = remix.remix(remixdirs[1],nStp)
            # gsphs[1].AddCPCP(nStp,AxM,xy=[0.610,0.925])
            mviz.AddIonBoxes(gs[0,3:6],ion1)

            ion2 = remix.remix(remixdirs[2],nStp)
            # gsphs[2].AddCPCP(nStp,AxR,xy=[0.610,0.925])
            mviz.AddIonBoxes(gs[0,6:],ion2)

        fOut = oDir+"/vid.%04d.png"%(npl)
        kv.savePic(fOut,bLenX=45)


if __name__ == "__main__":
    main()