import numpy as np

N = 51
Mat_rev = np.zeros([N,N])
Mat_test = np.zeros([2000, 8*N])

for ii in range(0,N):
    Mat_rev[ii,-ii-1] = 1

for ii in range(2000):
    for jj in range(8*N):
        Mat_test[ii,jj] = ii/ (jj+5)

y_res_list = []
for ii in range(4):
    C_tmp = Mat_test[:,ii*N:(ii+1)*N]@Mat_rev
    y_res_list.append(C_tmp)

for ii in range(4,8):
    q_tmp = Mat_test[:,ii*N:(ii+1)*N]@Mat_rev
    y_res_list.append(q_tmp)
y_res = np.concatenate(y_res_list, axis = 1)
print(y_res.shape)
'''
print(Mat_rev)
print('Before')
print(Mat_test)
print()
print('After')
print(Mat_test@Mat_rev)
'''

