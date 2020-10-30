import geocoder

g_maps_url = "http://maps.google.com/?q={},{}"
telegram='https://api.telegram.org/bot[botToken]/sendlocation?chat_id=[UserID]&latitude=51.6680&longitude=32.6546'

def get_current_location(g_maps_url):
    g = geocoder.ip('me')

    lat = g.latlng[0]
    lat = g.latlng[0] + 2.64

    long = g.latlng[1]
    long = g.latlng[1] + 1.3424

    print(lat, long)

    current_location =  g_maps_url.format(lat, long)

    return current_location

print(get_current_location(g_maps_url))