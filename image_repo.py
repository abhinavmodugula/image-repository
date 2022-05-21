from genericpath import exists
import click
import h5py
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from matplotlib import image, pyplot
from model import FeatureType, FeatureFactory
from config import hdf5_dir

"""
Command line tool to manage basic operations
in the image repository.

"""

#Helper Methods:
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    Credit: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def read_single_hdf5(image_id, model):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file.get("image")).astype("uint8")
    feature = np.array(file.get(model.name))
    #label = int(np.array(file["/meta"]).astype("uint8"))

    file.close()

    return image, feature

def read_many_hdf5(label, model):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{label}.h5", "r+")

    images = np.array(file.get("images")).astype("uint8")
    
    features = np.array(file.get(model.name))

    file.close()

    return images, features

#Command Line Methods:
@click.group("cli")
def cli():
    pass

@cli.command("add_image")
@click.argument('path', type=click.Path(exists=True))
@click.argument('label')
def add_image(path, label, model_type=FeatureType.GENERAL):
    im = image.imread(path)
    model = FeatureFactory.create(model_type)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{label}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(im), h5py.h5t.STD_U8BE, data=im
    )
    
    model_feature = np.asarray(model.describe(im))
    
    feature = file.create_dataset(
        model.name, np.shape(model_feature), h5py.h5t.IEEE_F32LE, data=model_feature
    )
    
    file.close()

@cli.command("add_collection")
@click.argument('directory')
@click.argument('label')
def add_many_images(directory, label, model_type=FeatureType.GENERAL):
    filelist = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    all_images = np.array([np.array(Image.open(fname)) for fname in filelist], dtype=object)
    num_files = len(filelist)

    filtered = len(all_images)
    default_size = all_images[0].shape[0]
    data_shape = all_images.shape + all_images[1].shape

    all_images = [image for image in all_images if image.shape[0] == default_size]
    filtered -= len(all_images)

    #TODO: Check if label already exists
    
    #calculate all features and concat them before adding to db
    model = FeatureFactory.create(model_type)
    printProgressBar(0, len(all_images), prefix = 'Calcuating features:', suffix = 'Complete', length = 50)
    features = []
    for i, image in enumerate(all_images):
        features.append(np.asarray(model.describe(image)))
        printProgressBar(i+1, len(all_images), prefix = 'Calcuating features:', suffix = 'Complete', length = 50)
    data = np.asarray(features)
    print("\n")
    
    printProgressBar(0, len(all_images), prefix = 'Converting to efficient file format:', suffix = 'Complete', length = 50)
    with h5py.File(hdf5_dir / f"{label}.h5", driver='core', backing_store=True, mode='w') as h5image:
        ds = h5image.create_dataset('images', shape=data_shape, dtype=str(all_images[0].dtype))
        feature = h5image.create_dataset(model.name, shape=np.shape(data), dtype=h5py.h5t.IEEE_F32LE)
        
        for cnt, image in enumerate(all_images):
            ds[cnt:cnt+1:,:,:] = image
            #feature[cnt:cnt+1,:] = np.asarray(model.describe(image))
            feature[cnt:cnt+1:,:] = data[cnt]
            printProgressBar(cnt + 1, len(all_images), prefix = 'Converting to efficient file format:', suffix = 'Complete', length = 50)
    
    print("\nNumber of images with wrong size: ")
    print(filtered)

    print("Added Successfully")

    #TODO: recusively call on filtered out images

@cli.command("list")
def list_collections():
    print("All collections:")
    directory = "images"
    filelist = [f for f in os.listdir(directory) if not os.path.isfile(f)]
    for file in filelist:
        print(file)

@cli.command("search")
@click.argument('query_path', type=click.Path(exists=True))
@click.option('--in', 'in_', type=str)
#For now, only searches in files with multiple images
def search(query_path, in_="all", model_type=FeatureType.GENERAL):
    model = FeatureFactory.create(model_type)
    collections = [f for f in os.listdir("data/") if ".h5" in f]

    query_image = image.imread(query_path)
    query_feature = model.describe(query_image)

    printProgressBar(0, len(collections), prefix = 'Loading Images:', suffix = 'Complete', length = 50)

    all_images = []
    all_features = []
    for i, collection in enumerate(collections):
        name = collection.split(".")[0]
        file = h5py.File(hdf5_dir/f"{name}.h5", "r+")
        if "images" in file.keys():
            images, features = read_many_hdf5(name, model)
            all_images.append(images)
            all_features.append(features)
        file.close()
        printProgressBar(i + 1, len(collections), prefix = 'Loading Images:', suffix = 'Complete', length = 50)
    
    if len(all_images) == 0:
        print("Add images to dataset before searching")
        return
    
    print("Performing search...")
    
    combined_images = all_images[0]
    combined_features = all_features[0]
    for i in range(1, len(all_images)):
        combined_images = np.concatenate((combined_images, all_images[i]), axis=0)
        combined_features = np.concatenate((combined_features, all_features[i]), axis=0)
    
    df = pd.DataFrame(combined_features)
    search_res = df.apply(lambda x : model.compare(query_feature, x), axis=1)

    N = 10
    most_similar = search_res.nsmallest(N).index

    print("Found results, now displaying")

    for res in most_similar:
        im = combined_images[res]
        pyplot.imshow(im)
        pyplot.show()

if __name__ == "__main__":
    cli()
    
