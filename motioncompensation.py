"""
Functions to estimate the flow between two images and compensate it.

DeepFlow seems to be much better than SimpleFlow in terms of warping quality,
and much faster than TVL1 in terms of speed
"""
import numpy as np
import cv2

# Parameters of the motion estimation algorithms
def warp_flow(img, flow):
	'''
		Applies to img the transformation described by flow.
	'''
	assert len(flow.shape) == 3 and flow.shape[-1] == 2

	hf, wf = flow.shape[:2]
	# flow 		= -flow
	flow[:, :, 0] += np.arange(wf)
	flow[:, :, 1] += np.arange(hf)[:, np.newaxis]
	res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
	return res

def estimate_invflow(img0, img1, me_algo):
	'''
		Estimates inverse optical flow by using the me_algo algorithm.
	'''
	# # # img0, img1 have to be uint8 grayscale
	assert img0.dtype == 'uint8' and img1.dtype == 'uint8'

	# Create estimator object
	if me_algo == "DeepFlow":
		of_estim = cv2.optflow.createOptFlow_DeepFlow()
	elif me_algo == "SimpleFlow":
		of_estim = cv2.optflow.createOptFlow_SimpleFlow()
	elif me_algo == "TVL1":
		of_estim = cv2.DualTVL1OpticalFlow_create()
	else:
		raise Exception("Incorrect motion estimation algorithm")

	# Run flow estimation (inverse flow)
	flow = of_estim.calc(img1, img0, None)
#	flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	return flow

def align_frames(img_to_align, img_source, mc_alg='DeepFlow'):
	'''
		Applies to img_to_align a transformation which converts it into img_source.

		Args:
			img_to_align: HxWxC image
			img_source: HxWxC image
			mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
		Returns:
			HxWxC aligned image
	'''

	# make sure images are uint8 in the [0, 255] range
	if img_source.max() <= 1.0:
		img_source = (img_source*255).clip(0, 255)
	img_source = img_source.astype(np.uint8)
	if img_to_align.max() <= 1.0:
		img_to_align = (img_to_align*255).clip(0, 255)
	img_to_align = img_to_align.astype(np.uint8)

	img0 = img_to_align[:, :, 0]
	img1 = img_source[:, :, 0]
	out_img = None

	# Align frames according to selection in mc_alg
	flow = estimate_invflow(img0, img1, mc_alg)

	# rectifier
	out_img = warp_flow(img_to_align, flow)

	return out_img

if __name__=="__main__":
	import glob
	import imageio 

	dir2imgs = "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p/car-race"
	list2imgs = sorted(glob.glob(dir2imgs+"/*jpg"))

	print("Total image number=", len(list2imgs))
	img0 = imageio.imread(list2imgs[0])/255.0
	img1 = imageio.imread(list2imgs[3])/255.0
	# flow = estimate_invflow(img0, img1, me_algo="SimpleFlow")
	warped_img0 = align_frames(img0, img1, mc_alg='DeepFlow')/255.0
	# print(np.max(warped_img0))
	diff_before_align = np.abs(img0-img1)
	diff_after_align = np.abs(warped_img0-img1)
	imageio.imwrite("img0_img1.jpg", np.hstack((img0, img1)))
	imageio.imwrite("img0_warpedImg0.jpg", np.hstack((img0, warped_img0)))
	imageio.imwrite("diff_beforeAfterAlign.jpg", np.hstack((diff_before_align, diff_after_align)))
