import sys
import numpy as np
import os
import scipy.sparse as sp

def save_sim(targetfile, sim_res, loop):
    print('sim_res.shape', sim_res.shape)
    density = 0
    if loop:
        with open(targetfile, 'w') as outfile:
            for i in range(sim_res.shape[0])[1:]:
                for j in range(sim_res.shape[1])[1:]:
                    if sim_res[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(round(sim_res[i][j], 2)) + '\n')
                        density += 1
    else:
        with open(targetfile, 'w') as outfile:
            for i in range(sim_res.shape[0])[1:]:
                for j in range(sim_res.shape[1])[1:]:
                    if sim_res[i][j] != 0:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(round(sim_res[i][j], 2)) + '\n')
                        density += 1
    density = density / (sim_res.shape[0]-1) / (sim_res.shape[1]-1)
    return density

def normalize_csr(M):
    row_sum = np.array(M.sum(1).flatten() / M.getnnz(1))
    row_sum[np.isinf(row_sum)] = 0.
    r_inv = np.power(row_sum, -1.0 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    col_sum = np.array(M.sum(0).flatten() / M.getnnz(0))
    col_sum[np.isinf(col_sum)] = 0.
    c_inv = np.power(col_sum, -1.0 / 2).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)

    return r_mat_inv.dot(M).dot(c_mat_inv)

def save_coo(data_coo, write_file):
    values = data_coo.data.reshape(-1, 1)
    rows = data_coo.row.reshape(-1, 1)
    columns = data_coo.col.reshape(-1, 1)
    data_np = np.hstack((rows, columns, values))
    np.savetxt(write_file, data_np, fmt=['%d', '%d', '%.4f'], delimiter=' ')
    M, N = data_coo.get_shape()[0], data_coo.get_shape()[1]
    density = len(values) / M / N
    print('density of %s = ' % write_file.split('/')[-1][:-4], density)


class meta_sim_generator:
    def __init__(self, data_dir):
        self.K = 500

        self.sim_dir = os.path.join(data_dir, 'sim_res')
        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
        
        user_list, item_list = [], []
        for line in open(os.path.join(data_dir, 'train.txt'), 'r').readlines():
            line = line.strip().split(' ')
            user = int(line[0])
            for i in line[1:]:
                user_list.append(user)
                item_list.append(int(i))
        ones = [1] * len(item_list)

        if 'Douban_Book' in data_dir:
            self.save_UB(ones, user_list, item_list, 'UB.dat')
            exit(0)
            ba = np.loadtxt(os.path.join(data_dir, 'ba.dat'), dtype=np.int)
            bp = np.loadtxt(os.path.join(data_dir, 'bp.dat'), dtype=np.int)
            by = np.loadtxt(os.path.join(data_dir, 'by.dat'), dtype=np.int)
            
            # ug = np.loadtxt(os.path.join(data_dir, 'ug.dat'), dtype=np.int)
            # uloc = np.loadtxt(os.path.join(data_dir, 'uloc.dat'), dtype=np.int)
            # uu = np.loadtxt(os.path.join(data_dir, 'uu.dat'), dtype=np.int)

            self.bnum = len(set(ba[:, 0]) | set(bp[:, 0]) | set(by[:, 0]))
            self.unum = len(set(user_list))
            self.anum = len(set(ba[:, 1]))
            self.pnum = len(set(bp[:, 1]))
            self.ynum = len(set(by[:, 1]))
            # self.gnum = len(set(ug[:, 1]))
            # self.locnum = len(set(uloc[:, 1]))
            with open(os.path.join(self.sim_dir, 'n_users_items.txt'), 'w') as writer:
                writer.write('%d %d\n'%(self.unum, self.bnum))
            
            ub_csr = sp.coo_matrix((ones, (user_list, item_list)), shape=(self.unum, self.bnum)).tocsr()
            ba_csr = sp.coo_matrix((np.ones(len(ba)), (ba[:, 0], ba[:, 1])), shape=(self.bnum, self.anum)).tocsr()
            bp_csr = sp.coo_matrix((np.ones(len(bp)), (bp[:, 0], bp[:, 1])), shape=(self.bnum, self.pnum)).tocsr()
            by_csr = sp.coo_matrix((np.ones(len(by)), (by[:, 0], by[:, 1])), shape=(self.bnum, self.ynum)).tocsr()

            # ug_csr = sp.coo_matrix((np.ones(len(ug)), (ug[:, 0], ug[:, 1])), shape=(self.unum, self.gnum)).tocsr()
            # uloc_csr = sp.coo_matrix((np.ones(len(uloc)), (uloc[:, 0], uloc[:, 1])), shape=(self.unum, self.locnum)).tocsr()
            # uu_csr = sp.coo_matrix((np.ones(len(uu)), (uu[:, 0], uu[:, 1])), shape=(self.unum, self.unum)).tocsr()

            self.get_collabrative_net(ub_csr, 'UBU.dat', 'BUB.dat', 'UBUB.dat')
            # self.get_UU_mode(uu_csr, ub_csr, 'UU.dat', 'UUB.dat')
            # self.get_UOU_mode(ug_csr, ub_csr, 'UGU.dat', 'UGUB.dat')
            # self.get_UOU_mode(uloc_csr, ub_csr, 'ulocu.dat', 'ulocub.dat')
            self.get_BOB_mode(ba_csr, ub_csr, 'BAB.dat', 'UBAB.dat')
            self.get_BOB_mode(bp_csr, ub_csr, 'BPubB.dat', 'UBPubB.dat')
            self.get_BOB_mode(by_csr, ub_csr, 'BYB.dat', 'UBYB.dat')
            # self.get_UBOBU_mode(ba_csr, ub_csr, 'UBABU.dat', 'UBABUB.dat')
           

        elif 'Douban_Movie' in data_dir:
            self.save_UB(ones, user_list, item_list, 'UM.dat')
            exit(0)
            ma = np.loadtxt(os.path.join(data_dir, 'ma.dat'), dtype=np.int)
            md = np.loadtxt(os.path.join(data_dir, 'md.dat'), dtype=np.int)
            mt = np.loadtxt(os.path.join(data_dir, 'mt.dat'), dtype=np.int)
            
            # ug = np.loadtxt(os.path.join(data_dir, 'ug.dat'), dtype=np.int)
            # uu = np.loadtxt(os.path.join(data_dir, 'uu.dat'), dtype=np.int)

            self.mnum = len(set(ma[:, 0]) | set(md[:, 0]) | set(mt[:, 0]))
            self.unum = len(set(user_list))
            self.anum = len(set(ma[:, 1]))
            self.dnum = len(set(md[:, 1]))
            self.tnum = len(set(mt[:, 1]))
            # self.gnum = len(set(ug[:, 1]))
            with open(os.path.join(self.sim_dir, 'n_users_items.txt'), 'w') as writer:
                writer.write('%d %d\n'%(self.unum, self.mnum))
            exit(0)

            um_csr = sp.coo_matrix((ones, (user_list, item_list)), shape=(self.unum, self.mnum)).tocsr()
            ma_csr = sp.coo_matrix((np.ones(len(ma)), (ma[:, 0], ma[:, 1])), shape=(self.mnum, self.anum)).tocsr()
            md_csr = sp.coo_matrix((np.ones(len(md)), (md[:, 0], md[:, 1])), shape=(self.mnum, self.dnum)).tocsr()
            mt_csr = sp.coo_matrix((np.ones(len(mt)), (mt[:, 0], mt[:, 1])), shape=(self.mnum, self.tnum)).tocsr()
            # ug_csr = sp.coo_matrix((np.ones(len(ug)), (ug[:, 0], ug[:, 1])), shape=(self.unum, self.gnum)).tocsr()
            # uu_csr = sp.coo_matrix((np.ones(len(uu)), (uu[:, 0], uu[:, 1])), shape=(self.unum, self.unum)).tocsr()

            self.get_collabrative_net(um_csr, 'UMU.dat', 'MUM.dat', 'UMUM.dat')
            # self.get_UU_mode(uu_csr, um_csr, 'UU.dat', 'UUM.dat')
            # self.get_UOU_mode(ug_csr, um_csr, 'UGU.dat', 'UGUM.dat')
            self.get_BOB_mode(ma_csr, um_csr,  'MAM.dat', 'UMAM.dat')
            self.get_BOB_mode(md_csr, um_csr, 'MDM.dat', 'UMDM.dat')
            self.get_BOB_mode(mt_csr, um_csr, 'MTM.dat', 'UMTM.dat')
            # self.get_UBOBU_mode(ma_csr, um_csr,  'UMAMU.dat', 'UMAMUM.dat')
            # self.get_UBOBU_mode(md_csr, um_csr, 'UMDMU.dat', 'UMDMUM.dat')
            # self.get_UBOBU_mode(mt_csr, um_csr, 'UMTMU.dat', 'UMTMUM.dat')
        
        elif 'Yelp_Business' in data_dir:
            self.save_UB(ones, user_list, item_list, 'UB.dat')
            exit(0)
            bca = np.loadtxt(os.path.join(data_dir, 'bca.dat'), dtype=np.int)
            bci = np.loadtxt(os.path.join(data_dir, 'bci.dat'), dtype=np.int)
            
            # ucomp = np.loadtxt(os.path.join(data_dir, 'ucomp.dat'), dtype=np.int)
            # uu = np.loadtxt(os.path.join(data_dir, 'uu.dat'), dtype=np.int)

            self.bnum = len(set(bca[:, 0]) | set(bci[:, 0]))
            self.unum = len(set(user_list))
            self.canum = len(set(bca[:, 1]))
            self.cinum = len(set(bci[:, 1]))
            # self.compnum = len(set(ucomp[:, 1]))
            with open(os.path.join(self.sim_dir, 'n_users_items.txt'), 'w') as writer:
                writer.write('%d %d\n'%(self.unum, self.bnum))
            exit(0)
            
            ub_csr = sp.coo_matrix((ones, (user_list, item_list)), shape=(self.unum, self.bnum)).tocsr()
            bca_csr = sp.coo_matrix((np.ones(len(bca)), (bca[:, 0], bca[:, 1])), shape=(self.bnum, self.canum)).tocsr()
            bci_csr = sp.coo_matrix((np.ones(len(bci)), (bci[:, 0], bci[:, 1])), shape=(self.bnum, self.cinum)).tocsr()
            # ucomp_csr = sp.coo_matrix((np.ones(len(ucomp)), (ucomp[:, 0], ucomp[:, 1])), shape=(self.unum, self.compnum)).tocsr()
            # uu_csr = sp.coo_matrix((np.ones(len(uu)), (uu[:, 0], uu[:, 1])), shape=(self.unum, self.unum)).tocsr()

            self.get_collabrative_net(ub_csr, 'UBU.dat', 'BUB.dat', 'UBUB.dat')
            # self.get_UU_mode(uu_csr, ub_csr, 'UU.dat', 'UUB.dat')
            # self.get_UOU_mode(ucomp_csr, ub_csr, 'UCompU.dat', 'UCompUB.dat')
            self.get_BOB_mode(bca_csr, ub_csr, 'BCaB.dat', 'UBCaB.dat')
            self.get_BOB_mode(bci_csr, ub_csr, 'BCiB.dat', 'UBCiB.dat')
            # self.get_UBOBU_mode(bca_csr, ub_csr, 'UBCaBU.dat', 'UBCaBUB.dat')
            # self.get_UBOBU_mode(bci_csr, ub_csr, 'UBCiBU.dat', 'UBCiBUB.dat')

    def save_UB(self, ones, user_list, item_list, write_file):
        ones = np.array(ones)
        user_list = np.array(user_list)
        item_list = np.array(item_list)
        ui = np.vstack((user_list, item_list))
        triplets = np.vstack((ui, ones)).T
        print(triplets.shape)
        np.savetxt(os.path.join(self.sim_dir, write_file), triplets, fmt=['%d', '%d', '%.4f'], delimiter=' ')


    def get_collabrative_net(self, ub_csr, UIU_file, IUI_file, UIUI_file):
        ubu_csr = ub_csr.dot(ub_csr.transpose())
        bub_csr = ub_csr.transpose().dot(ub_csr)
        ubub_csr = ubu_csr.dot(ub_csr)

        ubu_csr = normalize_csr(ubu_csr)
        bub_csr = normalize_csr(bub_csr)
        ubub_csr = normalize_csr(ubub_csr)

        save_topK_items(ubu_csr, os.path.join(self.sim_dir, UIU_file), self.K)
        save_topK_items(bub_csr, os.path.join(self.sim_dir, IUI_file), self.K)
        save_topK_items(ubub_csr, os.path.join(self.sim_dir, UIUI_file), self.K)        
        
        '''
        ubu_coo = ubu_csr.tocoo()
        bub_coo = bub_csr.tocoo()
        ubub_coo = ubub_csr.tocoo()

        save_coo(ubu_coo, os.path.join(self.sim_dir, UIU_file))
        save_coo(bub_coo, os.path.join(self.sim_dir, IUI_file))
        save_coo(ubub_coo, os.path.join(self.sim_dir, UIUI_file))
        '''


    def get_UU_mode(self, uu_csr, ub_csr, UU_file, UUI_file):
        uub_csr = uu_csr.dot(ub_csr)
        uub_csr = normalize_csr(uub_csr)

        save_topK_items(uub_csr, os.path.join(self.sim_dir,  UUI_file))
        '''
        uub_coo = uub_csr.tocoo()
        save_coo(uub_coo, os.path.join(self.sim_dir, UUI_file))
        '''
        uu_coo = uu_csr.tocoo()
        save_coo(uu_coo, os.path.join(self.sim_dir, UU_file))


    def get_UOU_mode(self, uo_csr, ub_csr, UOU_file, UOUB_file):
        uou_csr = uo_csr.dot(uo_csr.transpose())
        uoub_csr = uou_csr.dot(ub_csr)

        uou_csr = normalize_csr(uou_csr)
        uoub_csr = normalize_csr(uoub_csr)

        save_topK_items(uou_csr, os.path.join(self.sim_dir, UOU_file))
        save_topK_items(uoub_csr, os.path.join(self.sim_dir, UOUB_file))
        '''
        uou_coo = uou_csr.tocoo()
        uoub_coo = uoub_csr.tocoo()

        save_coo(uou_coo, os.path.join(self.sim_dir, UOU_file))
        save_coo(uoub_coo, os.path.join(self.sim_dir, UOUB_file))
        '''

    def get_BOB_mode(self, bo_csr, ub_csr, BOB_file, UBOB_file):
        bob_csr = bo_csr.dot(bo_csr.transpose())
        ubob_csr = ub_csr.dot(bob_csr)

        bob_csr = normalize_csr(bob_csr)
        ubob_csr = normalize_csr(ubob_csr)

        save_topK_items(bob_csr, os.path.join(self.sim_dir, BOB_file))
        save_topK_items(ubob_csr, os.path.join(self.sim_dir, UBOB_file))
        '''
        bob_coo = bob_csr.tocoo()
        ubob_coo = ubob_csr.tocoo()

        save_coo(bob_coo, os.path.join(self.sim_dir, BOB_file))
        save_coo(ubob_coo, os.path.join(self.sim_dir, UBOB_file))
        '''


    def get_UBOBU_mode(self, bo_csr, ub_csr, UBOBU_file, UBOBUB_file):
        bob_csr = bo_csr.dot(bo_csr.transpose())
        ubobu_csr = ub_csr.dot(bob_csr).dot(ub_csr.transpose())
        ubobub_csr = ubobu_csr.dot(ub_csr)

        ubobu_csr = normalize_csr(ubobu_csr)
        ubobub_csr = normalize_csr(ubobub_csr)

        save_topK_items(ubobu_csr, os.path.join(self.sim_dir, UBOBU_file))
        save_topK_items(ubobub_csr, os.path.join(self.sim_dir, UBOBUB_file))
        '''
        ubobu_coo = ubobu_csr.tocoo()
        ubobub_coo = ubobub_csr.tocoo()

        save_coo(ubobu_coo, os.path.join(self.sim_dir, UBOBU_file))
        save_coo(ubobub_coo, os.path.join(self.sim_dir, UBOBUB_file))
        '''


def save_topK_items(data_csr, write_file, topK=500):
    M, N = data_csr.get_shape()
    triplets = []
    for i in range(M):
        items = data_csr.getrow(i).toarray().flatten()
        cols = np.argpartition(-items, topK).flatten()[:topK]  # descending order
        cols = [c for c in cols if items[c] > 0]
        triplets.extend([(i, c, items[c]) for c in cols])
    triplets = np.array(triplets)
    density = len(triplets) / M / N
    np.savetxt(write_file, triplets, fmt=['%d', '%d', '%.4f'], delimiter=' ')
    print('density of %s = ' % write_file.split('/')[-1][:-4], density)





  
if __name__ == '__main__':
    dataset = 'Douban_Movie'
    if 'Douban_Book' in dataset:  
        # 只是相乘， 看不出 UMAMUM 是分解为 U-MAM-U-M  还是 U-MA-MUM  第二种没有意义？
        metapath = ['UBUB', 'UBAB', 'UBPubB', 'UBYB']  # , 'UBPubBUB',  'UBYBUB' , 'UUB', 'UGUB', 'UBABUB'
    elif 'Douban_Movie' in dataset:
        metapath = ['UMUM', 'UMDM', 'UMAM ', 'UMTM']  # 'UGUM' 
    elif 'Yelp' in dataset:
        metapath = ['UBUB',  'UBCaB', 'UBCiB']  # 'UBCiBU'  'UUB', 'UCompUB',
    
    data_dir = '../processed_data/' + dataset

    meta_sim_generator(data_dir)

