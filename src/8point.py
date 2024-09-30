# https://gist.github.com/jensenb/8668000              # decomp E into R, t
# https://github.com/Smelton01/8-point-algorithm.git   # original source, images

#%% 
import numpy as np
from matplotlib import pyplot as plt

# %%
# low depth variance image same plane -> coplanar = degernate solution
# point-pairs in close1 and close2 : [x_close1, y_close1, x_close2, y_close2]
uvMat0 = [[1855,1352, 1680, 1305], [2100, 2116, 1914, 2084], [2482, 1994, 2310, 1984], [3070, 1336, 2959, 1337], [3450, 1911, 3369, 1966], [1356, 1885, 1173, 1809],\
    [3415,2125, 3320, 2190], [3848, 1308, 3846, 1342]]

# low depth variance different plane
# point-pairs in close1 and close2 : [x_close1, y_close1, x_close2, y_close2]
uvMat1 = [[1855,1352, 1680, 1305], [1251,2640, 960, 2518], [525, 2436, 323, 2273], [745, 1840, 611,1734], [1578, 2890, 1174, 2783], [1356, 1885, 1173, 1809],\
    [3415,2125, 3320, 2190], [3848, 1308, 3846, 1342]]

# high depth variance image same plane -> coplanar = degernate solution
# point-pairs in far1 and far2 : [x_far1, y_far1, x_far2, y_far2]
uvMat2 = [[580, 2362, 492, 1803], [2050, 2097, 1381, 1956], [2558, 2174, 1544, 2115], [1395, 1970, 1166, 1752], [2490, 3003, 466, 2440], [3368, 1622, 3320, 2011],\
    [2183, 1500, 2471,1621], [1972,1775, 1674, 1736]]

# high depth variance image different plane
# point-pairs in far1 and far2 : [x_far1, y_far1, x_far2, y_far2]
uvMat3 = [[580, 2362, 492, 1803],[3316,1276, 3242, 1565], [1007,788, 1606,885] , [1900, 1144, 2330, 1250], [984, 1369, 1574, 1335], [3368, 1622, 3320, 2011],\
    [2192, 1288, 2469, 1420], [2050, 2097, 1381, 1956]]

#%%

close1 = plt.imread("../res/same1.jpg")
close2 = plt.imread('../res/same2.jpg')
far1   = plt.imread("../res/extreme1.jpg")
far2   = plt.imread('../res/extreme2.jpg')

# assume camera-matrix is known as:
img_width  = close1.shape[1]
img_height = close1.shape[0]
f = 1000. # focal length
pu = img_width / 2
pv = img_height / 2
K = np.array([(f, 0, pu),
              (0, f, pv),
              (0, 0,  1)])
K_inv = np.linalg.inv(K)

# %%
# compute essential matrix E
def calc_E(uvMat, K):
    A = np.zeros((len(uvMat),9))
    K_inv = np.linalg.inv(K)
    
    for i in range(len(uvMat)):
        uv1 = np.array([uvMat[i][0], uvMat[i][1], 1.0])
        x1_norm = K_inv.dot( uv1 )
        uv2 = np.array([uvMat[i][2], uvMat[i][3], 1.0])
        x2_norm = K_inv.dot( uv2 )
        A[i][0] = x1_norm[0]*x2_norm[0] # x1*x2
        A[i][1] = x1_norm[1]*x2_norm[0] # y1*x2
        A[i][2] = x2_norm[0]            #    x2
        A[i][3] = x1_norm[0]*x2_norm[1] # x1*y2
        A[i][4] = x1_norm[1]*x2_norm[1] # y1*y2
        A[i][5] = x2_norm[1]            #    y2
        A[i][6] = x1_norm[0]            # x1
        A[i][7] = x1_norm[1]            # y1
        A[i][8] = 1.0                   # 1.0
    
    _,_,Vt = np.linalg.svd(A) # returns U,S,Vt
    
    E_vec = Vt.transpose()[:,8] # minimal solution: chose eigenvector of smallest eigenvalue
    E = E_vec.reshape((3,3))

    return E

#%%
plt.imshow(close1)
for pt_pair in uvMat0:
    x = pt_pair[0]
    y = pt_pair[1]
    plt.plot(x, y, "rx")
for pt_pair in uvMat1:
    x = pt_pair[0]
    y = pt_pair[1]
    plt.plot(x, y, "gx")
plt.title('close1')
plt.show()

plt.imshow(close2)
for pt_pair in uvMat0:
    x = pt_pair[2]
    y = pt_pair[3]
    plt.plot(x, y, "rx")
for pt_pair in uvMat1:
    x = pt_pair[2]
    y = pt_pair[3]
    plt.plot(x, y, "gx")
plt.title('close2')
plt.show()

plt.imshow(far1)
for pt_pair in uvMat2:
    x = pt_pair[0]
    y = pt_pair[1]
    plt.plot(x, y, "rx")
for pt_pair in uvMat3:
    x = pt_pair[0]
    y = pt_pair[1]
    plt.plot(x, y, "gx")
plt.title('far1')
plt.show()

plt.imshow(far2)
for pt_pair in uvMat2:
    x = pt_pair[2]
    y = pt_pair[3]
    plt.plot(x, y, "rx")
for pt_pair in uvMat3:
    x = pt_pair[2]
    y = pt_pair[3]
    plt.plot(x, y, "gx")
plt.title('far2')
plt.show()

#%%
cmap = plt.get_cmap("jet_r")

# plot on img1
def plot1(img, E, points):
    F = K_inv.T @ E @ K_inv
    
    w = img.shape[1]
    num = 1
    for x, y, *pt in points:
        color = cmap(num/len(points))
        a,b,c = np.array([*pt, 1]).transpose() @ F
        p1 = (0,-c/b)
        p2 = (w, -(a*w + c)/b)
        plt.plot(*zip(p1,p2), color=color)
        plt.plot(x, y, "x",color=color)
        plt.text(x,y,f"{num}")
        num+=1
    plt.imshow(img)

    plt.show()

# plot on img 2 
# epipolar lines based on  x'y' and points x,y
def plot2(img, E, points):
    F = K_inv.T @ E @ K_inv
    
    w = img.shape[1]
    num = 1
    for *pt, x,y in points:
        color = cmap(num/len(points))
        a,b,c = F @ [*pt, 1]
        p1 = (0,-c/b)
        p2 = (w, -(a*w + c)/b)
        plt.plot(*zip(p1,p2), color=color)
        plt.plot(x,y, "x", color=color)
        plt.text(x,y,f"{num}")
        num+=1
    plt.imshow(img)
    
    plt.show()

#%%

if False:
    #uvMat = uvMat0 # degenerate solution for close image
    uvMat = uvMat1  # correct solution for close image
    E = calc_E(uvMat, K)
    
    # Plot the epipolar lines
    plot1(close1, E, uvMat) 
    plot2(close2, E, uvMat)
else:
    #uvMat = uvMat2 # degenerate solution for far image
    uvMat = uvMat3  # correct solution for far image
    E = calc_E(uvMat, K)

    # Plot the epipolar lines
    plot1(far1, E, uvMat) 
    plot2(far2, E, uvMat)

# %%

# compute epipolar points
def compute_epipoles(f_mat):
    f_mat_2 = f_mat.transpose() @ f_mat
    w3, v3 = np.linalg.eig(f_mat_2)
    smallest = np.argmin(w3)
    f_vec = v3[:,smallest]
    e1_vec = f_vec/f_vec[2]
    with np.printoptions(precision=1, suppress=True):
        print("e1: ", e1_vec)
    # p1 = uvMat[0]
    # point1 = p1[2:]
    # print(f_mat @ e1vec @ [*point1,1])

    f_T_f_mat = f_mat @ f_mat.transpose()
    w4 , v4 = np.linalg.eig(f_T_f_mat)
    smallest = np.argmin(w4)
    f_vec = v4[:,smallest]
    e2_vec = f_vec/f_vec[2]
    with np.printoptions(precision=1, suppress=True):
        print("e2: ", e2_vec)
    # point2 = p1[:2]
    # print(e2vec @ f_mat @ [*point2,1]) 

compute_epipoles(E)

# %%

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

# compute t and R from essential matrix E
def decompose_E(E, uvMat):
    # enforce rank(E) = 2
    # (gives just minor changes)
    # U,s,Vt = np.linalg.svd(E)
    # set smallest eigenvalue s[2] to zero:
    # s_rank2 = np.diag([s[0], s[1], 0])
    # E_rank2 = U @ s_rank2 @ Vt

    # convert pixel coordinates to normalized camera coordinates
    pts_cam1 = []
    pts_cam2 = []
    for i in range(len(uvMat)):
        uv1 = np.array([uvMat[i][0], uvMat[i][1], 1.0])
        x1_norm = K_inv.dot( uv1 )
        pts_cam1.append(x1_norm)
        
        uv2 = np.array([uvMat[i][2], uvMat[i][3], 1.0])
        x2_norm = K_inv.dot( uv2 )
        pts_cam2.append(x2_norm)

    # decompose essential matrix into R, t (see Hartley and Zisserman 9.13)
    U, s, Vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    
    # only in one of the four configurations will all the points be in front of both cameras
    # First choice: R = U * W * Vt, t = +u_3 (see Hartley Zisserman 9.19)
    R = U @ W @ Vt
    t = U[:, 2]
    
    if not in_front_of_both_cameras(pts_cam1, pts_cam2, R, t):
        # Second choice: R = U * W * Vt, t = -u_3
        t = - U[:, 2]
        
        if not in_front_of_both_cameras(pts_cam1, pts_cam2, R, t):
            # Third choice: R = U * Wt * Vt, t = u_3
            R = U.dot(W.T).dot(Vt)
            t = U[:, 2]
    
            if not in_front_of_both_cameras(pts_cam1, pts_cam2, R, t):
                # Fourth choice: R = U * Wt * Vt, t = -u_3
                t = - U[:, 2]
    
    return R, t

R, t = decompose_E(E, uvMat)
   
