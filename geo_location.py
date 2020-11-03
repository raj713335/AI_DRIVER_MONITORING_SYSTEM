import geocoder

# g_maps_url = "http://maps.google.com/?q={},{}"
# telegram='https://api.telegram.org/bot[botToken]/sendlocation?chat_id=[UserID]&latitude=51.6680&longitude=32.6546'
#
# def get_current_location(g_maps_url):
#     g = geocoder.ip('me')
#
#     lat = g.latlng[0]
#     lat = g.latlng[0] + 2.64
#
#     long = g.latlng[1]
#     long = g.latlng[1] + 1.3424
#
#     print(lat, long)
#
#     current_location =  g_maps_url.format(lat, long)
#
#     return current_location
#
# print(get_current_location(g_maps_url))


# import requests
# import json
#
# send_url = 'http://freegeoip.net/json'
# r = requests.get(send_url)
# j = json.loads(r.text)
# lat = j['latitude']
# lon = j['longitude']
#
# print(lat,lon)



import requests

# Step 1) Find the public IP of the user. This is easier said that done, look into the library Netifaces if you're
# interested in getting the public IP locally.
# The GeoIP API I'm going to use here is 'https://geojs.io/' but any service offering similar JSON data will work.

ip_request = requests.get('https://get.geojs.io/v1/ip.json')
my_ip = ip_request.json()['ip']  # ip_request.json() => {ip: 'XXX.XXX.XX.X'}
print(my_ip)
# Prints The IP string, ex: 198.975.33.4

# Step 2) Look up the GeoIP information from a database for the user's ip

geo_request_url = 'https://get.geojs.io/v1/ip/geo/' + my_ip + '.json'
geo_request = requests.get(geo_request_url)
geo_data = geo_request.json()
print(geo_data)
# {
# "area_code": "0",
# "continent_code": "NA",
# "country": "United States",
# "country_code": "US",
# "country_code3": "USA",
# "ip": "198.975.33.4",
# "latitude": "37.7510",
# "longitude": "-97.8220",
# "organization": "AS15169 Google Inc.",
# "timezone": ""
# }  This is a fake example I grabbed from the GeoJS website

lat = geo_data['latitude']
lon = geo_data['longitude']

print(lat,lon)