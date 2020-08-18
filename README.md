# cmip6-albedo

![Build status][image-1]
![CodeBeat][image-2]
![CodeCov][image-3]

# Total irradiance calculations
It is possible to add more accurate models for extra terrestrial light using various models when 
calculating the following:
```python
dni_extra = pvlib.irradiance.get_extra_radiation(time)

total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                            surface_azimuth,
                                                            apparent_zenith,
                                                            azimuth,
                                                            irrads['dni'],
                                                            irrads['ghi'],
                                                            irrads['dhi'],
             dni_extra=dni_extra_array,
             model='haydavies')
```

If you need the angle of incidence:
```python
aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                                      solpos['apparent_zenith'].to_numpy(), 	   solpos['azimuth'].to_numpy())
```
To reference the use of pvlib for light calculations use:
Holmgren, W., C. Hansen and M. Mikofski (2018). “pvlib Python: A python package for modeling solar energy systems.” 
Journal of Open Source Software 3(29): 884.

### Useful links
http://www.matteodefelice.name/post/aggregating-gridded-data/
https://cds.climate.copernicus.eu/toolbox/doc/index.html
https://www.toptal.com/python/an-introduction-to-mocking-in-python
https://esmtools.readthedocs.io/en/latest/examples/pco2.html
earthsystemmodeling.org/esmf\_releases/last\_built/esmpy\_doc/html/examples.html
https://github.com/Quick/Nimble#truthiness

\#

[image-1]:	https://badge.buildkite.com/998b597662a8db957ab524d2660958105de691cc0bc1753594.svg
[image-2]:	https://codebeat.co/badges/8bf4f052-6579-47fa-a552-b221154549c0
[image-3]:	https://codecov.io/gh/trondkr/cmip6-albedo/branch/master/graph/badge.svg