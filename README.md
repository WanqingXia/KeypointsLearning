# KeypointsLearning
1. run calc_box to calculate the object's bounding box diagonal length with
python calc_box.py
2. run generate.py to sample camera points and generate images with
blenderproc run generate.py
if the previous command does not work, try python ./../cli.py run generate_ycb.py
3. run calc_angles.py to calculate the angles between a point and it's neigbouring points with
python calc_angles.py
4. run sift.py to calculate and filter 2D keypoints
5. convert .obj to .ply with meshlabserver -i models/021_bleach_cleanser/textured.obj -o bottle.ply
6. view output file python ./../cli.py vis hdf5 output/0.hdf5
