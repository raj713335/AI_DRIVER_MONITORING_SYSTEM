import geocoder

g_maps_url = "http://maps.google.com/?q={},{}"

def get_current_location(g_maps_url):
    g = geocoder.ip('me')
    #lat = g.latlng[0] + 2.64
    lat = g.latlng[0]
    #long = g.latlng[1] + 1.3424
    long = g.latlng[1]
    #print(lat, long)
    current_location =  g_maps_url.format(lat, long)
    return current_location

get_current_location(g_maps_url)