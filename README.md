# image-repository
Tool to maintain and search a large number of images. Currently, it has the ability to add a large number of images and also search for an image that is similar to one provided.

## Storing Images
It was important for the tool to be capable of handling many (> 100k) images in a fast and efficient way. This is achieved using the HDF5 file format to store images instead of just the raw files.

Added a directory of images:
```
python image_repo.py add_collection /images/vacation vacation_may_19
```

![Command Line Demo of Adding Images](https://media.giphy.com/media/VsyvTjeLbZRm6SgDwK/giphy.gif)

## Features and Search
Searching for related images was implemented through color histograms as described here:
https://pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/

Other features can be easily added by adding it to the model.py file. Each feature needs to just have a desribe and a comapre method as follows:
```
class Feature:
  
  def describe(image):
    pass
  
  def compare(feature1, feature2):
    pass
```

That combined with the optimizations from using HDF5 creates a usuable tool to search for similar images.

```
python image_repo.py search query_image.png
```
