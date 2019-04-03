import pandas as pd 
import numpy as np

#this script processes the Schich Freebase Dataset
#reading in the data
df = pd.read_csv('SchichDataS1_FB copy.csv', sep = ',')

# creating 'movement' indicator, indicates whether individual has moved from place of birth
df['Movement'] = np.where(df['BLocLabel'] == df['DLocLabel'], 0, 1)


#creating 'other' indicator
df['Other'] = np.where((df['PerformingArts'] == 0) & 
               (df['Sports'] == 0) & 
               (df['Business/Industry/Travel'] == 0) & 
               (df['Gov/Law/Mil/Act/Rel'] == 0) &
               (df['Creative'] == 0) &
               (df['Academic/Edu/Health'] == 0), 1, 0)

# creating decade column
df['Decade'] = np.floor(np.floor(df['BYear'])/10)

#creating movement interaction variables

df['Move_PA'] = np.where((df['PerformingArts'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_Sports'] = np.where((df['Sports'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_BIT'] = np.where((df['Business/Industry/Travel'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_GLMAR'] = np.where((df['Gov/Law/Mil/Act/Rel'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_C'] = np.where((df['Creative'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_AEH'] = np.where((df['Academic/Edu/Health'] == 1) & (df['Movement'] == 1), 1, 0)
df['Move_Other'] = np.where((df['Other'] == 1) & (df['Movement'] == 1), 1, 0)

#seperating data into location and decade, then counting the number of people who have 'moved' during their life at that location at that decade
#we can figure out immigrants by grouping on death location, emigrants by birth location
city_decade_df_im = df.groupby(['DLocLabel', 'Decade']).sum().reset_index()
city_decade_df_em = df.groupby(['BLocLabel', 'Decade']).sum().reset_index() #same but for emigrants

#creating 'Number Born' and 'Number Died' Categories

city_decade_df_im['no_Died'] = df.groupby(['DLocLabel', 'Decade']).count().reset_index()['PrsID']
city_decade_df_em['no_Born'] = df.groupby(['BLocLabel', 'Decade']).count().reset_index()['PrsID']

#renaming columns of immigration
city_decade_df_im.rename(columns = {'Movement': 'Immigrants', 
                                    'DLocLabel': 'place_upper',
                                    'Move_C' : 'Im_Creative', 
                                    'Move_PA': 'Im_PA', 
                                    'Move_Sports': 'Im_Sports', 
                                    'Move_BIT': 'Im_BIT', 
                                    'Move_GLMAR' : 'Im_GLMAR',
                                    'Move_AEH' : 'Im_AEH',
                                    'Move_Other': 'Im_Other'}, inplace = True)
city_decade_df_em.rename(columns = {'Movement':'Emigrants',
                                    'BLocLabel': 'place_upper',
                                    'Move_C' : 'Em_Creative', 
                                    'Move_PA': 'Em_PA', 
                                    'Move_Sports': 'Em_Sports', 
                                    'Move_BIT': 'Em_BIT', 
                                    'Move_GLMAR' : 'Em_GLMAR',
                                    'Move_AEH' : 'Em_AEH',
                                    'Move_Other' : 'Em_Other'}, inplace = True)

#renaming columns of born/death

city_decade_df_im.rename(columns = {'Movement': 'Immigrants', 
                                    'DLocLabel': 'place_upper',
                                    'Creative' : 'Died_Creative', 
                                    'PerformingArts': 'Died_PA', 
                                    'Sports': 'Died_Sports', 
                                    'Business/Industry/Travel': 'Died_BIT', 
                                    'Gov/Law/Mil/Act/Rel' : 'Died_GLMAR',
                                    'Academic/Edu/Health' : 'Died_AEH',
                                    'Other': 'Died_Other'}, inplace = True)
city_decade_df_em.rename(columns = {'Movement':'Emigrants',
                                    'BLocLabel': 'place_upper',
                                    'Creative' : 'Born_Creative', 
                                    'PerformingArts': 'Born_PA', 
                                    'Sports': 'Born_Sports', 
                                    'Business/Industry/Travel': 'Born_BIT', 
                                    'Gov/Law/Mil/Act/Rel' : 'Born_GLMAR',
                                    'Academic/Edu/Health' : 'Born_AEH',
                                    'Other': 'Born_Other'}, inplace = True)

#making each city name uppercase strings to standardize formatting
city_decade_df_im['place_upper'] = city_decade_df_im['place_upper'].str.upper()
city_decade_df_em['place_upper'] = city_decade_df_em['place_upper'].str.upper()

#merging of immigrants and emigrants datasets -- if no imigrant/emigrant data is available for observation, we have NaN

city_decade_df_im['Emigrants'] = np.nan
city_decade_df_em['Immigrants'] = np.nan

city_decade_df_im = city_decade_df_im.drop_duplicates(['place_upper', 'Decade'])
city_decade_df_em = city_decade_df_em.drop_duplicates(['place_upper', 'Decade'])


#merging the two dataframes
city_decade_df = pd.merge(city_decade_df_im , city_decade_df_em, on = ['place_upper', 'Decade'], how = 'outer')

#dealing with NAN lattitude and longtitudes

longx_list = list(city_decade_df['DLocLong_x'])
longy_list = list(city_decade_df['DLocLong_y'])
latx_list = list(city_decade_df['DLocLat_x'])
laty_list = list(city_decade_df['DLocLat_y'])

#filling all NANs with 0
city_decade_df = city_decade_df.fillna(0) 

for lattitude in longx_list:
    if lattitude == 0:
        latx_list[longx_list.index(lattitude)] = laty_list[laty_list.index(lattitude)]
    else:
        continue
        
for longtitude in longx_list:
    if longtitude == 0:
        longx_list[longx_list.index(longtitude)] = longy_list[longy_list.index(longtitude)]
    else:
        continue
        
        
city_decade_df['Longtitude'] = longx_list
city_decade_df['Lattitude'] = latx_list

#immigrants/emigrants clean up
    
city_decade_df['Immigrants'] = city_decade_df['Immigrants_x']
city_decade_df['Emigrants'] = city_decade_df['Emigrants_y']


#dropping irrelevant variables left over from the merging function

for name in list(city_decade_df):
    if name.endswith(('_x', '_y')):
       city_decade_df =  city_decade_df.drop(name, axis = 1)   

# making decades the same as premerge dataframe

city_decade_df['Decade'] = city_decade_df['Decade'].apply(lambda x: x*10)

#Ensuring city/decades in each dataset have the same longtitude/lattitudes

#All the cities in city_decade will have the same longtitudes and latittudes
unique_city = list(city_decade_df['place_upper'].unique())

for city in unique_city:
    city_decade_df[city_decade_df['place_upper'] == city]['Longtitude'] = city_decade_df[city_decade_df['place_upper'] == city]['Longtitude'].iloc[0]
    city_decade_df[city_decade_df['place_upper'] == city]['Lattitude'] = city_decade_df[city_decade_df['place_upper'] == city]['Lattitude'].iloc[0]

#Using the opencage geocoder to obtain lattitudes/longtitudes from location names
from opencage.geocoder import OpenCageGeocode

key = '1c967b863ce042be8589f5fdb3f2cfe0'
geocoder = OpenCageGeocode(key)

#function for obtaining lattitudes and longtitudes from opencage
def geocode_lat_long(location):
    lat = geocoder.geocode(location)[0]['geometry']['lat']
    long = geocoder.geocode(location)[0]['geometry']['lng']
    return([lat, long])

def geocode_city(lng, lat):
    city = geocoder.reverse_geocode(lat, lng)[0]['components']['city']
    return(city)

#replacing longtitudes and lattitudes for locations that can be geocoded
#if the loop encounters an error we simply skip the location

unique_city = list(city_decade_df['place_upper'].unique())

for city in unique_city:
    try:
        #taking coordinates and returning appropriate city if it exits
        lattitude = list(city_decade_df.loc[city_decade_df.place_upper == city, 'Lattitude'])[0]
        longtitude = list(city_decade_df.loc[city_decade_df.place_upper == city, 'Longtitude'])[0]
        actual_city = geocode_city(longtitude, lattitude)
        #changing observation name to the city implied by the coordinates
        city_decade_df.loc[city_decade_df.place_upper == city, 'place_upper'] = actual_city.upper() #uppercase formatting
        
        #finding the correct geocoded coordinates for the city found above
        y = geocode_lat_long(actual_city)
        actual_lattitude = y[0]
        actual_longtitude = y[1]
        city_decade_df.loc[city_decade_df.place_upper == city, 'Lattitude'] = actual_lattitude
        city_decade_df.loc[city_decade_df.place_upper == city, 'Longtitude'] = actual_longtitude
    except:
        try:
            #if longtitudes and lattitudes need to be swapped
            lattitude = list(city_decade_df.loc[city_decade_df.place_upper == city, 'Lattitude'])[0]
            longtitude = list(city_decade_df.loc[city_decade_df.place_upper == city, 'Longtitude'])[0]
            actual_city = geocode_city(lattitude, longtitude)
            city_decade_df.loc[city_decade_df.place_upper == city, 'place_upper'] = actual_city.upper() #uppercase formatting
            y = geocode_lat_long(actual_city)
            actual_lattitude = y[0]
            actual_longtitude = y[1]
            city_decade_df.loc[city_decade_df.place_upper == city, 'Lattitude'] = actual_lattitude
            city_decade_df.loc[city_decade_df.place_upper == city, 'Longtitude'] = actual_longtitude
        except:
            # there are 88 cities that need to be skipped, we omit these from the dataset
            continue
        
# Originally, we ran the first 'try loop' and then appended a list of cities that were throwing errors
# we had a look manually and found that the longtitude and lattitude of some of these cities needed to be swapped
# we ran another iteration of the 'try loop' and were left with 88 cities that contained errors, in general these were not 'cities' and
# were specific places such as 'site of 2010 polish plane crash' -- we ommitted these from the datset

city_decade_df.to_csv('geocode_cdd_new.csv', sep = ',')

#there are 88 observations omitted from the finished product, these seem to be very specific places and not particularly important