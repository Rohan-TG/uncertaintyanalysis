import os
import h5py
import openmc.data
import urllib.request

# url = 'https://anl.box.com/shared/static/kxm7s57z3xgfbeq29h54n7q6js8rd11c.ace'
# filename, headers = urllib.request.urlretrieve(url, 'gd157.ace')
h1 = openmc.data.IncidentNeutorn.from_endf('n-001_H_001_test1.endf')
# # Load ACE data into object
# gd157 = openmc.data.IncidentNeutron.from_ace('gd157.ace')


