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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',help='PESSTO Marshall .csv file',default=None)
    parser.add_argument('-a','--agecut',default=None)
    return parser.parse_args()



class marshall_list():

    def __init__(self,df,site= 'La Silla',newax=True,count=0):
        self.df = df
        self._set_observatory(site)
        if site == 'La Silla':
            self.tz = 'America/Santiago'
        self._set_tonight()
        if newax:
            self._plot_base()
        self.counter=count


    def _set_observatory(self,site):
        '''Sets the observatory site'''
        if site == 'La Silla':
            self.loc = EarthLocation.of_site('La Silla Observatory')
        else:
            self.loc = EarthLocation.of_site(site)
    def _set_tonight(self):
        '''
        Takes date in format dd/mm/yy and sets the night as starting on the date before
        arguments: date (str)
        '''
        timezone = pytz.timezone(self.tz)
        utcoffset = datetime.now(timezone).utcoffset()
        midnight = Time(datetime.combine(datetime.now(timezone).date(), time(0, 0)) + timedelta(1)-utcoffset)
        self.midnight = midnight
        delta_midnight = np.linspace(-8, 10, 1000)*u.hour
        self.tonight = AltAz(obstime=midnight+delta_midnight,
                                  location=self.loc)
    def _plot_base(self):
        f,ax = plt.subplots(figsize=(10,6))


        # Sun! ##############################
        t = self.tonight.obstime.value
        ax.hlines(20,t[0],t[-1],linestyle='--',color='white',linewidth=3)
        sunaltazs_tonight = get_sun(self.tonight.obstime).transform_to(self.tonight)
        self.plot_vis_obj(sunaltazs_tonight,ax =ax,
                     linestyle='dashed',label = 'Sun',color='orange',linewidth=10,zorder=6)

        plt.fill_between(t, 0, 90,
                         sunaltazs_tonight.alt < -0*u.deg, color='midnightblue', zorder=0)
        plt.fill_between(t, 0, 90,
                         sunaltazs_tonight.alt < -18*u.deg, color='k', zorder=0)
        plt.fill_between(t, 0, 90,
                         sunaltazs_tonight.alt >= 0*u.deg, color='deepskyblue', zorder=0)
        ####################################

        xloc = ax.xaxis.set_major_locator(HourLocator(byhour=range(0,24,1)))
        hours_fmt = DateFormatter('%H')
        ax.xaxis.set_major_formatter(hours_fmt)

        # LEGEND


        # Axes Limits and Labels
        plt.grid(color='w',alpha=0.7)
        ax.set_xlabel('Time (UT)',fontsize=16)
        ax.set_ylim(0,90)
        ax.set_ylabel('Altitude (deg)',fontsize=16)
        plt.tick_params(which='both',direction='inout')
        plt.title('NTT Visibility ' + datetime.strftime(sunaltazs_tonight.obstime[0].to_datetime(),"%Y-%m-%d"),
                                                       fontsize=18)
        fontP = FontProperties()
        fontP.set_size('small')
        ax.legend(title="Objects", prop=fontP,loc='upper center', bbox_to_anchor=(0.9, 1),
                  ncol=1, fancybox=True, shadow=True)

    def set_time(self,date='today'):
        '''Sets the time to now'''
        if date == 'today':
            self.time_now = Time.now()

    def set_night(self,date=None ):
        '''
        Takes date in format dd/mm/yy and sets the night as starting on the date before
        arguments: date (str)
        '''
        timezone = pytz.timezone(self.tz)
        utcoffset = datetime.now(timezone).utcoffset()
        if not date:
            midnight = datetime.combine(datetime.now(timezone).date(), time(0, 0)) + timedelta(1) + utcoffset
        else:
            midnight = datetime.strptime(date,'%d/%m/%y')
        delta_midnight = np.linspace(-8, 10, 1000)*u.hour
        self.tonight = AltAz(obstime=midnight+delta_midnight,
                                  location=self.loc)
        # to add later
    def set_site(self,site):
        try:
            EarthLocation.of_site(site)
        except:
            raise Exception("Site not in Astropy database. Try EarthLocation.get_site_names()")


    def cut_old_detections(self,age=7):
        '''Cuts objects first detected more than a certain time ago
        parameters: date (str)
        '''

        newdate = datetime.now()-timedelta(age)
        self.df  = self.df[pd.to_datetime(self.df['discovery date'])>newdate]

    def get_obj_altaz(self,ra,dec):
        obj_coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
        obj_altazs= obj_coord.transform_to(self.tonight)
        return obj_altazs

    def plot_vis_obj(self,obj_altazs,ax=None,**kwargs):
        if not ax:
            ax = plt.gca()
        ax.plot_date(obj_altazs.obstime.datetime,
                     obj_altazs.alt,
                     marker=None,**kwargs)

    def plot_priority_classify(self,ax=None):
        if not ax:
            ax = plt.gca()
        null_patches = []
        priority_map = {
            'HIGH':{'c':'g','ls':'-','lw':3,'alpha':0.9},
            'MEDIUM':{'c':'y','ls':'--','lw':2,'alpha':0.7},
            'LOW':{'c':'r','ls':':','lw':1,'alpha':0.7}
        }
        for counter,i in enumerate(self.df.sort_values(by='discovery date',ascending=False).index):
            counter= counter+self.counter

            obj = self.df.loc[i]

            print('Doing %s'%obj['name'],counter)
            priority = obj['priority']
            altaz = self.get_obj_altaz(obj['ra'],obj['dec'])
            self.plot_vis_obj(altaz,
                            ax=ax,
                             **priority_map[priority])

            # Annotate the plot with a number corresponding to the current counter
            xy = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max())
            xytext = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max()-3)
            ax.annotate(counter+1,xy=xy,xytext=xytext,color='w',size=9)
            null_patches.append(mpatches.Patch(color='white',
                                label = str(counter+1) + ': '+obj['name'],
                                               ))
        fontP = FontProperties()
        fontP.set_size('small')
        l = plt.legend(handles=null_patches,title="Classifications", prop=fontP,loc='upper center', bbox_to_anchor=(0.9, 1),
                  ncol=1, fancybox=True, shadow=True)
        ax.add_artist(l)
        self.counter=counter+1
        print('Finished plotting %s objects'%self.counter)

    def plot_namelist(self,list,ax=None):
        if not ax:
            ax = plt.gca()
        null_patches = []
        priority_map = {
            'HIGH':{'c':'g','ls':'-','lw':3,'alpha':0.9},
            'MEDIUM':{'c':'y','ls':'--','lw':2,'alpha':0.7},
            'LOW':{'c':'r','ls':':','lw':1,'alpha':0.7}
        }
        for counter,objname in enumerate(list):
            counter= counter+self.counter

            obj = self.df.loc[self.df[self.df['name']==objname].index]

            print('Doing %s'%obj['name'].values[0],counter)
            priority = obj['priority'].values[0]
            altaz = self.get_obj_altaz(obj['ra'].values[0],obj['dec'].values[0]) # Want to make this neater
            self.plot_vis_obj(altaz,
                            ax=ax,
                             **priority_map[priority])

            # Annotate the plot with a number corresponding to the current counter
            xy = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max())
            xytext = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max()-3)
            ax.annotate(counter+1,xy=xy,xytext=xytext,color='w',size=9)
            null_patches.append(mpatches.Patch(color='white',
                                label = str(counter+1) + ': '+obj['name'].values[0],
                                               ))
        fontP = FontProperties()
        fontP.set_size('small')
        l = plt.legend(handles=null_patches,title="Classifications", prop=fontP,loc='upper center', bbox_to_anchor=(1.05, 0.7),
                  ncol=1, fancybox=True, shadow=True)
        ax.add_artist(l)
        self.counter=counter+1
        print('Finished plotting %s objects'%self.counter)

    def plot_priority_followup(self,ax=None,priority='ALL'):
        '''
        arguments:
        ax: axis instance
        priority: list of priorities to include
        '''
        if not ax:
            ax = plt.gca()
        null_patches = []
        priority_map = {
            'CRITICAL':{'c':'g','ls':'-'},
            'IMPORTANT':{'c':'y','ls':'--'},
            'USEFUL':{'c':'r','ls':':'},
            'NONE':{'c':'w','ls':':'},
        }
        if priority !='ALL':
            has_low_priority = self.df[self.df['priority']!=priority].index
            self.df.drop(rows = has_low_priority,inplace=True)
        for counter,i in enumerate(self.df.sort_values(by='discovery date',ascending=False).index):
            counter = counter+self.counter
            obj = self.df.loc[i]

            print('Doing %s'%obj['name'],counter)
            priority = obj['priority']
            altaz = self.get_obj_altaz(obj['ra'],obj['dec'])
            self.plot_vis_obj(altaz,
                            ax=ax,
                             color=priority_map[priority]['c'],
                             linestyle=priority_map[priority]['ls'])

            # Annotate the plot with a number corresponding to the current counter
            xy = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max())
            xytext = (mdates.date2num(altaz.obstime[altaz.alt.to_value().argmax()].to_datetime()),
                altaz.alt.to_value().max()+1)
            ax.annotate(counter+1,xy=xy,xytext=xytext,color='w',size=9)
            null_patches.append(mpatches.Patch(color='white',
                                label = str(counter+1) + ': '+obj['name'],
                                               ))

        fontP = FontProperties()
        fontP.set_size('small')
        l = plt.legend(handles=null_patches,title='Follow-up', prop=fontP,loc='upper left', bbox_to_anchor=(-0.32, 1),
                  ncol=1, fancybox=True, shadow=True)
        ax.add_artist(l)
        self.counter=counter
        print('Finished plotting %s objects'%self.counter)


def main():
    args = parser()

    marshall_df  = pd.read_csv(args.filename)

    plt.show()

if __name__ =="__main__":
    main()
