Requirements
============

- Basically we're looking to build a system like eCognition, for segmenting, processing and classifying geospatial imagery (satellite/airborne)
- We're basing it on what eCognition was like around 2008-2011, not necessarily what it is like now
- We need a full GUI - this could be inside something like a QGIS plugin or a standalone UI. It needs to work on Windows, Mac and Linux
- We need a comprehensive Python interface, and ideally a command line interface
- We could base it on tools and features available in RSGISLib or other Python libraries, or we could build things ourselves
- We need to have the ability to 1) segment an image (with configurable segment 'resolution' or size), then 2) extract information for each segment (geometry info like size and area, statistical measures of band values like mean, stdev etc, texture values etc) and then 3) use these for classification or export to other systems
- It should be able to support multiple segmentation algorithms - eg. the Shepherd algorithm described in one of the papers, Meta's Segment Anything Model etc
- Classification should be able to be via various algorithms (eg. from sklearn), either supervised or unsupervised
- Somehow it should work nicely with embedding data
- It should be easy to use for non-experts and easy to install on all platforms

