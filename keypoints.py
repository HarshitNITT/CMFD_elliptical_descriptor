import numpy as np
import numpy.linalg as LA

def potentialKeypointDetection(pyramidLayer, w=16):
	potentialKeypoint = []

	
	pyramidLayer[:,:,0] = 0
	pyramidLayer[:,:,-1] = 0
	
	for i in range(w//2+1, pyramidLayer.shape[0]-w//2-1):
		for j in range(w//2+1, pyramidLayer.shape[1]-w//2-1):
			for k in range(1, pyramidLayer.shape[2]-1): 
				patch = pyramidLayer[i-1:i+2, j-1:j+2, k-1:k+2]
				#here the central point will have index 13
				
				if np.argmax(patch) == 13 or np.argmin(patch) == 13:
					potentialKeypoint.append([i, j, k])

	return potentialKeypoint

def localizingKeypoint(pyramidLayer, x, y, s):
	dx = (pyramidLayer[y,x+1,s]-pyramidLayer[y,x-1,s])/2.
	dy = (pyramidLayer[y+1,x,s]-pyramidLayer[y-1,x,s])/2.
	ds = (pyramidLayer[y,x,s+1]-pyramidLayer[y,x,s-1])/2.

	dxx = pyramidLayer[y,x+1,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x-1,s]
	dxy = ((pyramidLayer[y+1,x+1,s]-pyramidLayer[y+1,x-1,s]) - (pyramidLayer[y-1,x+1,s]-pyramidLayer[y-1,x-1,s]))/4.
	dxs = ((pyramidLayer[y,x+1,s+1]-pyramidLayer[y,x-1,s+1]) - (pyramidLayer[y,x+1,s-1]-pyramidLayer[y,x-1,s-1]))/4.
	dyy = pyramidLayer[y+1,x,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y-1,x,s]
	dys = ((pyramidLayer[y+1,x,s+1]-pyramidLayer[y-1,x,s+1]) - (pyramidLayer[y+1,x,s-1]-pyramidLayer[y-1,x,s-1]))/4.
	dss = pyramidLayer[y,x,s+1]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x,s-1]

	J = np.array([dx, dy, ds])
	HD = np.array([
		[dxx, dxy, dxs],
		[dxy, dyy, dys],
		[dxs, dys, dss]])
	
	offset = -LA.inv(HD).dot(J)	
	return offset, J, HD[:2,:2], x, y, s

def getPotentialKeypoints(pyramidLayer, R_th, t_c, w):
	potentialKeypoint = potentialKeypointDetection(pyramidLayer, w)
	#print('%d candidate keypoints found' % len(potentialKeypoint))

	keypoints = []

	for i, cand in enumerate(potentialKeypoint):
		y, x, s = cand[0], cand[1], cand[2]
		offset, J, H, x, y, s = localizingKeypoint(pyramidLayer, x, y, s)

		contrast = pyramidLayer[y,x,s] + .5*J.dot(offset)
		if abs(contrast) < t_c: continue

		w, v = LA.eig(H)
		r = w[1]/w[0]
		R = (r+1)**2 / r
		if R > R_th: continue

		kp = np.array([x, y, s]) + offset
		if kp[1] >= pyramidLayer.shape[0] or kp[0] >= pyramidLayer.shape[1]: continue # throw out boundary point

		keypoints.append(kp)

	#print('%d keypoints found' % len(keypoints))
	return np.array(keypoints)

def get_keypoints(DOG_Pyramid, R_th, t_c, w):
    finalKyepoints = []

    for pyramidLayer in DOG_Pyramid:
        finalKyepoints.append(getPotentialKeypoints(pyramidLayer, R_th, t_c, w))

    return finalKyepoints
