import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.dates import HourLocator, DateFormatter
from matplotlib.font_manager import FontProperties
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import astropy.io.fits as fits
from astropy.table import Table
import seaborn as sns
from datetime import datetime,timedelta, time
import pytz
import itertools
import argparse
import os

plt.style.use('seaborn-colorblind')
sns.set_color_codes(palette='colorblind')
fname = input('marshall csv file path: ')
marshall = pd.read_csv(fname)



def vis_curve(i,ra,dec,counter,**kwargs):
    obj_coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    midnight = Time('2019-7-30 00:00:00') - utcoffset
    delta_midnight = np.linspace(-8, 10, 1000)*u.hour
    tonight = AltAz(obstime=midnight+delta_midnight,
                              location=LS)
    obj_altazs= obj_coord.transform_to(tonight)

    ax.plot_date(obj_altazs.obstime.datetime,
                 obj_altazs.alt,
                 marker=None,**kwargs)


    return(obj_altazs)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

priority_map = {
    'HIGH':{'c':'g','ls':'-'},
    'MEDIUM':{'c':'c','ls':'--'},
    'LOW':{'c':'red','ls':':'}
}

plt.style.use('seaborn-colorblind')
sns.set_context('notebook')
f,ax = plt.subplots(figsize=(10,6))
#LS = EarthLocation(lat=(70*u.deg+44*u.arcmin+1.5*u.arcsec), lon=(-1*(29*u.deg + 15*u.arcmin +32.1*u.arcsec)), height=2375*u.m)
LS = EarthLocation.of_site('La Silla Observatory')
utcoffset = -4*u.hour  # Eastern Daylight Time
timezone = 'America/Santiago'
time = Time(datetime.combine(datetime.now(timezone).date(), time(0, 0))) - utcoffset
newdate = datetime.strptime("20/07/2019", "%d/%m/%Y")
'''n=0

for i in marshall.index:
    discdate = datetime.strptime(marshall['discovery date'].loc[i],"%Y-%m-%d")

    if  discdate> newdate:
        n+=1
palette = itertools.cycle(sns.color_palette(palette='rainbow',n_colors=n))'''
null_patches=[]
for counter,i in enumerate(marshall.sort_values(by='discovery date',ascending=False).index):
    discdate = datetime.strptime(marshall['discovery date'].loc[i],"%Y-%m-%d")

    if  discdate> newdate:
        #c=next(palette)
        print('Doing %s'%marshall['name'].loc[i])
        #t = vis_curve(i,marshall['ra'].loc[i],marshall['dec'].loc[i]).to_value()
        priority = marshall['priority'].loc[i]
        altaz = vis_curve(i,marshall['ra'].loc[i],marshall['dec'].loc[i],
                          counter,
                          color=priority_map[priority]['c'],
                          linestyle=priority_map[priority]['ls'],)
        xy = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
            altaz.alt.to_value().max())
        print(xy)
        xytext = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
            altaz.alt.to_value().max()+1)
        ax.annotate(counter+1,xy=xy,xytext=xytext,color='w',size=9)
        null_patches.append(mpatches.Patch(color='white',
                            label = str(counter+1) + ': '+marshall['name'].loc[i],
                                           ))
t = altaz.obstime.value
ax.hlines(20,t[0],t[-1],linestyle='--',color='white',linewidth=3)

from astropy.coordinates import get_sun
delta_midnight = np.linspace(-12, 12, 1000)*u.hour

# Sun! ##############################
frame_tonight = AltAz(obstime=altaz.obstime, location=LS)
sunaltazs_tonight = get_sun(altaz.obstime).transform_to(frame_tonight)
ax.plot_date(sunaltazs_tonight.obstime.datetime,sunaltazs_tonight.alt,
             linestyle='dashed',marker=None,label = 'Sun',color='yellow',linewidth=10)

plt.fill_between(t, 0, 90,
                 sunaltazs_tonight.alt < -0*u.deg, color='midnightblue', zorder=0)
plt.fill_between(t, 0, 90,
                 sunaltazs_tonight.alt < -18*u.deg, color='k', zorder=0)
plt.fill_between(t, 0, 90,
                 sunaltazs_tonight.alt >= 0*u.deg, color='deepskyblue', zorder=0)
####################################
from matplotlib.dates import HourLocator, DateFormatter

xloc = ax.xaxis.set_major_locator(HourLocator(byhour=range(0,24,2)))
hours_fmt = DateFormatter('%H')
ax.xaxis.set_major_formatter(hours_fmt)

# LEGEND
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')


ax.legend(handles=null_patches,title="Objects", prop=fontP,loc='upper center', bbox_to_anchor=(0.95, 1.2),
          ncol=1, fancybox=True, shadow=True)

# Axes Limits and Labels
plt.grid(color='w',alpha=0.7)
ax.set_xlabel('Time (UT)',fontsize=16)
ax.set_ylim(0,90)
ax.set_ylabel('Altitude (deg)',fontsize=16)
plt.tick_params(which='both',direction='inout')
plt.title('NTT Visibility ' + datetime.strftime(sunaltazs_tonight.obstime[0].to_datetime(),"%Y-%m-%d"),
                                               fontsize=18)
plt.show()
plt.close()
savename = os.path.join(os.curdir(),'NTT_visibility'+now.strftime(format='%y-%m-%dT%H.%M'+'.png'))
plt.savefig(savename)
