#!/usr/bin/python3

import os;
import sys;
import dicom;
import matplotlib.pyplot as plt;
import numpy as np;
from scipy import ndimage;
from skimage import measure;
from mpl_toolkits.mplot3d.art3d import Poly3DCollection;
from plotly import figure_factory as FF;
from plotly.offline import iplot;

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

def make_mesh(images, threshold = -300, step_size = 1):

  p = np.transpose(np.array(images), (2,1,0)); # shape = (width, height, batch)
  verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size = step_size, allow_degenerate = True);
  return verts, faces;

def plot3d(images, threshold = -300, step_size = 1):

  verts, faces = make_mesh(images, threshold, step_size);
  x,y,z = zip(*verts);
  fig = plt.figure(figsize = (10,10));
  ax = fig.add_subplot(111, projection = '3d');
  mesh = Poly3DCollection(verts[faces], linewidths = 0.05, alpha = 1);
  face_color = [1,1,0.9];
  mesh.set_facecolor(face_color);
  ax.add_collection3d(mesh);
  ax.set_xlim(0, max(x));
  ax.set_ylim(0, max(y));
  ax.set_zlim(0, max(z));
  ax.set_facecolor((0.7,0.7,0.7));
  plt.show();

def plot3d_interactive(images, threshold = -300, step_size = 1):

  verts, faces = make_mesh(images, threshold, step_size);
  x,y,z = zip(*verts);
  colormap = ['rgb(236,236,212)','rgb(236,236,212)'];
  fig = FF.create_trisurf(x = x,y = y,z = z,plot_edges = False, colormap = colormap, simplices = faces, backgroundcolor = 'rgb(64,64,64)', title = 'Interactive Visualization');
  iplot(fig);

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <directory>");
    exit(1);
  # load slices from directory
  patient = load_scan(sys.argv[1]);
  # convert to hu
  imgs = get_pixels_hu(patient);
  show_stack(imgs);
  # change resolution
  resized_imgs = [change_resolution(img, patient, [1,1,1])[0] for img in imgs];
  show_stack(resized_imgs);
  # plot a static 3d image (it takes several minutes)
  plot3d(imgs);
  # plot a dynamic 3d image (it takes several minutes)
  plot3d_interactive(imgs);


