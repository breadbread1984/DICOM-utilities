#!/usr/bin/python3

import os;
import sys;
import dicom;
import matplotlib.pyplot as plt;
import numpy as np;
from scipy import ndimage;

def load_scan(dir):

  assert True == os.path.isdir(dir);
  slices = [dicom.read_file(os.path.join(dir, f)) for f in os.listdir(dir) if f.endswith('.dcm')];
  slices.sort(key = lambda x: int(x.InstanceNumber));
  try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]);
  except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation);
  for s in slices:
    s.SliceThickness = slice_thickness;
  return slices;

def get_pixels_hu(scans):

  image = np.stack([s.pixel_array for s in scans]);
  image = image.astype(np.int16);
  image[image == -2000] = 0;
  intercept = scans[0].RescaleIntercept;
  slop = scans[0].RescaleSlope;
  if slop != 1:
    image = slope * image.astype(np.float64);
    image = image.astype(np.int16);
  image += np.int16(intercept);
  # substance   HU
  # Air         -1000
  # Lung        -500
  # Fat         -100 to -50
  # Water       0
  # Blood       +30 to +70
  # Muscle      +10 to +40
  # Liver       +40 to +60
  # Bone        +700 to +3000
  return np.array(image, dtype = np.int16);

def show_stack(stack, rows = 6, cols = 6, sample_from = 10, sample_stride = 3):

  fig, ax = plt.subplots(rows, cols, figsize = [12,12]);
  for i in range(rows * cols):
    ind = sample_from + i * sample_stride;
    ax[i//rows,i%rows].set_title('slice %d' % ind);
    ax[i//rows,i%rows].imshow(stack[ind],cmap = 'gray');
    ax[i//rows,i%rows].axis('off');
  plt.show();

def change_resolution(image, scan, new_spacing = [1,1,1]):

  # change resolution so that new voxel has volume given in new_spacing
  # get voxel volume (slice thickness, pixel row spacing, pixel col spacing)
  spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype = np.float32);

  resize_factor = spacing / np.array(new_spacing, dtype = np.float32);
  new_resolution = np.round(image.shape * resize_factor[1:]);
  real_resize_factor = new_resolution / image.shape;
  new_spacing[1:] = spacing[1:] / real_resize_factor;

  image = ndimage.interpolation.zoom(image, real_resize_factor);

  return image, new_spacing;

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <directory>");
    exit(1);
  patient = load_scan(sys.argv[1]);
  imgs = get_pixels_hu(patient);
  show_stack(imgs);
  resized_imgs = [change_resolution(img, patient, [1,1,1])[0] for img in imgs];
  show_stack(resized_imgs);
