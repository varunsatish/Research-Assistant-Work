import pandas as pd

#reading the data
premerge = pd.read_csv('editions5geo_1650_to_merge.csv')
city_decade_df = pd.read_csv('geocode_cdd_new.csv', sep = ',')

#making uppercase
premerge['place_upper'] = premerge['place_upper'].str.upper()

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

#Ensuring city/decades in each dataset have the same longtitude/lattitudes

#All the cities in city_decade will have the same longtitudes and latittudes
unique_city = list(city_decade_df['place_upper'].unique())
unique_city_pre = list(premerge['place_upper'].unique())

#for each city in premerge, if it is in city_decade, we make the longtitudes and latitudes the same
unique_city = city_decade_df['place_upper'].unique()

#merging permerge and city_decade

new_merge_new = pd.merge(city_decade_df, premerge, right_on = ['place_upper', 'year10'], left_on = ['place_upper', 'Decade'], how = 'right')

#Using the opencage geocoder to obtain lattitudes/longtitudes from locations in new_merge
from opencage.geocoder import OpenCageGeocode

key = '1c967b863ce042be8589f5fdb3f2cfe0'
geocoder = OpenCageGeocode(key)

#function for obtaining lattitudes and longtitudes from opencage
def geocode_lat_long(location):
    lat = geocoder.geocode(location)[0]['geometry']['lat']
    lng = geocoder.geocode(location)[0]['geometry']['lng']
    return([lat, lng])

#geocoding the merged dataset

newlat = list(new_merge_new['lat_books'])
newlong = list(new_merge_new['lon_books'])

unique_city = new_merge_new['place_upper'].unique()

for city in unique_city:
    try:
        y = geocode_lat_long(city)
        lattitude = y[0]
        longtitude = y[1]
        newlat[city.index()] = lattitude
        newlong[city.index()] =  longtitude
        new_merge_new.loc[new_merge_new.place_upper == city, 'Longtitude'] = newlong
        new_merge_new.loc[new_merge_new.place_upper == city, 'Lattitude'] = newlat
    except:
        continue



#dropping irrelevant variables left over from the merging function

#we have a problem where there is a city/decade observation for all dummy variables
# we need to:
# [i] group by city_decade adn aggregate dummies
# [ii] ensure all other variables are not aggregated

#creating a datframe that aggregates all variables 

sum_df = new_merge_new.groupby(['place_upper', 'Decade']).sum().reset_index()

#creating a dataframe that takes the first observation of each unique city/decade observation
#used to create dataframe columns of those variables which need not be summed

df_no_sum = df2.groupby(['place_upper', 'Decade']).agg({
                                                         #'number of' variables
                                                         'n' : lambda x: x.iloc[0],
                                                         'nrel': lambda x: x.iloc[0],
                                                         'nsci': lambda x: x.iloc[0],
                                                         'narts': lambda x: x.iloc[0],
                                                         'nsocscience': lambda x: x.iloc[0],
                                                         'nother': lambda x: x.iloc[0],
                                                         'nforbidden': lambda x: x.iloc[0],
                                                         'npub10': lambda x: x.iloc[0],
                                                         'forbidden':  lambda x: x.iloc[0],
                                                         'no_Born': lambda x: x.iloc[0],
                                                         'no_Died': lambda x: x.iloc[0],
                                                         #indicator dummies
                                                         'Bohemia_and_Moravia': lambda x: x.iloc[0],
                                                         'Denmark': lambda x: x.iloc[0],
                                                         'England': lambda x: x.iloc[0],
                                                         'France': lambda x: x.iloc[0],
                                                         'Holy_Roman_Empire': lambda x: x.iloc[0],
                                                         'Hungary': lambda x: x.iloc[0],
                                                         'Italian_States': lambda x: x.iloc[0],
                                                         'Low_Countries': lambda x: x.iloc[0],
                                                         'Mexico': lambda x: x.iloc[0],
                                                         'Poland': lambda x: x.iloc[0],
                                                         'Portugal': lambda x: x.iloc[0],
                                                         'Scotland': lambda x: x.iloc[0],
                                                         'Spain': lambda x: x.iloc[0],
                                                         'Swiss_Confederation': lambda x: x.iloc[0],
                                                         'Longtitude': lambda x: x.iloc[0],
                                                         'Lattitude': lambda x: x.iloc[0]}).reset_index()

                                                      
                                                      
no_sum_cols = ['Died_PA', 'n', 'nrel', 'nsci', 'narts', 'nsocscience', 'nother', 'nforbidden',
     'npub10', 'forbidden', 'no_Born', 'no_Died','Bohemia_and_Moravia', 'Denmark',
     'England', 'France', 'Holy_Roman_Empire', 'Hungary', 'Italian_States', 'Low_Countries', 
     'Mexico', 'Poland', 'Portugal', 'Scotland', 'Spain', 'Swiss_Confederation', 'Longtitude',
     'Lattitude'] 

#replacing summed df with non-summed values

sum_df[no_sum_cols] = df_no_sum[no_sum_cols]

new_merge_new = sum_df


new_merge_new.to_csv('new_merge_new.csv', sep = ',')