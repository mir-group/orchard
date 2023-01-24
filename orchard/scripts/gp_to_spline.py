import yaml
import numpy as np
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import filter_cubic, eval_cubic
from ciderpress.dft.xc_models import NormGPFunctional
from sklearn.gaussian_process.kernels import RBF
from itertools import combinations
from argparse import ArgumentParser
from joblib import load, dump
from orchard.scripts.train_gp import parse_dataset
from ciderpress.models.kernels import *
import pyscf.lib

"""
Script for mapping a CIDER GP to a cubic spline.
Requires an input DFTGPR object, stored in joblib format.
"""

def get_dim(x, length_scale, density = 6, buff = 0.0, bound = None, max_ngrid = None):
    print(length_scale, bound)
    mini = np.min(x) - buff
    maxi = np.max(x) + buff
    if bound is not None:
        mini, maxi = bound[0], bound[1]
    ran = maxi - mini
    ngrid = max(int(density * ran / length_scale) + 1, 3)
    if max_ngrid is not None and ngrid > max_ngrid:
        ngrid = max_ngrid
    return (mini, maxi, ngrid)


def get_mapped_gp_evaluator_simple(gpr, rbf_density=8, max_ngrid=120):
    """
    Quick approach for pure RBF GP mapping for N <= 4
    """
    if gpr.args.agpr:
        raise ValueError('Must not be additive GP!')
    X = gpr.X
    alpha = gpr.gp.alpha_
    if gpr.args.use_ex_kernel:
        alpha *= X[:,0]
    D = X[:,1:]
    y = gpr.y
    N = D.shape[1]
    rbf = gpr.gp.kernel_.k1
    length_scale = rbf.k2.length_scale
    scale = np.array([rbf.k1.constant_value]) # TODO ??
    dims = []
    for i in range(N):
        dims.append(get_dim(D[:,i], length_scale[i],
                    density=rbf_density, bound=gpr.feature_list[i].bounds,
                    max_ngrid=max_ngrid))
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(N)]

    k0s = []
    print(length_scale)
    for i in range(D.shape[1]):
        print(length_scale[i], np.min(D[:,i]), np.max(D[:,i]))
        diff = (D[:,i:i+1] - grid[i][np.newaxis,:]) / length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))

    inds = list(np.arange(N))
    funcps = []
    spline_grids = []
    ind_sets = []
    if len(inds) == 0:
        raise ValueError('Kernel is constant!')
    elif len(inds) == 1:
        funcps.append(np.dot(alpha, k0s[inds[0]]))
        spline_grids.append(UCGrid(dims[inds[0]]))
        ind_sets.append(tuple(inds))
    elif len(inds) == 2:
        k = np.einsum('ni,nj->nij', k0s[inds[0]], k0s[inds[1]])
        spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]]))
        funcps.append(np.einsum('n,nij->ij', alpha, k))
        ind_sets.append(tuple(inds))
    elif len(inds) == 3:
        fps = 0
        for p0, p1 in pyscf.lib.prange(0, len(k0s[inds[0]]), 200):
            k = np.einsum(
                'ni,nj,nk->nijk',
                k0s[inds[0]][p0:p1],
                k0s[inds[1]][p0:p1],
                k0s[inds[2]][p0:p1]
            )
            fps += np.einsum('n,nijk->ijk', alpha[p0:p1], k)
        funcps.append(fps)
        spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]],
                                   dims[inds[2]]))
        ind_sets.append(tuple(inds))
    elif len(inds) == 4:
        k = np.einsum('ni,nj,nk,nl->nijkl', k0s[inds[0]], k0s[inds[1]],
                      k0s[inds[2]], k0s[inds[3]])
        funcps.append(np.einsum('n,nijkl->ijkl', alpha, k))
        spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]],
                                   dims[inds[2]], dims[inds[3]]))
        ind_sets.append(tuple(inds))
    else:
        raise ValueError('Order too high!')

    coeff_sets = [filter_cubic(spline_grids[0], funcps[0])]
    evaluator = NormGPFunctional(scale, ind_sets, spline_grids, coeff_sets,
                                 gpr.xed_y_converter, gpr.feature_list,
                                 gpr.desc_order, const=0.0,
                                 desc_version=gpr.desc_version,
                                 a0=gpr.a0, fac_mul=gpr.fac_mul,
                                 amin=gpr.amin)
    if gpr.args.use_ex_kernel:
        from ciderpress.models.gp import XED_Y_CONVERTERS
        evaluator.xed_y_converter = XED_Y_CONVERTERS['CHACHIYO']
    
    return evaluator

def get_mapped_gp_evaluator(gpr, test_x=None, test_y=None, test_rho_data=None,
                            srbf_density=8, arbf_density=8, max_ngrid=120):
    if not gpr.args.agpr:
        raise ValueError('Must be additive GP!')
    X = gpr.X
    alpha = gpr.gp.alpha_
    if gpr.args.use_ex_kernel:
        alpha *= X[:,0]
    D = X[:,1:]
    y = gpr.y
    n = gpr.args.agpr_nsingle
    N = D.shape[1]
    print(gpr.gp.kernel_)
    dims = []
    if n == 0:
        arbf = gpr.gp.kernel_.k1
        srbf = None
        ndim, length_scale, scale, order = arbf_args(arbf)
    elif n == 1:
        arbf = gpr.gp.kernel_.k1.k2
        srbf = gpr.gp.kernel_.k1.k1
        ndim, length_scale, scale, order = arbf_args(arbf)
        length_scale = np.append(srbf.length_scale, length_scale)
        dims = [get_dim(D[:,0], length_scale[0], density=srbf_density,
                        bound=gpr.feature_list[0].bounds,
                        max_ngrid=max_ngrid)]
    elif n > 1:
        arbf = gpr.gp.kernel_.k1.k2
        srbf = gpr.gp.kernel_.k1.k1
        ndim, length_scale, scale, order = arbf_args(arbf)
        length_scale = np.append(srbf.length_scale, length_scale)
        for i in range(n):
            dims.append(get_dim(D[:,i], length_scale[i], density=srbf_density,
                                bound=gpr.feature_list[i].bounds,
                                max_ngrid=max_ngrid))
    scale = np.array(scale)
    for i in range(n,N):
        dims.append(get_dim(D[:,i], length_scale[i],
                    density=arbf_density, bound=gpr.feature_list[i].bounds,
                    max_ngrid=max_ngrid))
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(N)]

    k0s = []
    print(length_scale)
    for i in range(D.shape[1]):
        print(length_scale[i], np.min(D[:,i]), np.max(D[:,i]))
        diff = (D[:,i:i+1] - grid[i][np.newaxis,:]) / length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    funcps = []
    spline_grids = []
    ind_sets = []
    if srbf is not None:
        srbf_inds = np.arange(n).tolist()
    arbf_inds = np.arange(n, N).tolist()
    assert order + n <= 4, 'Max order too high, must be at most 4'
    const = 0
    for o in range(order+1):
        for inds in combinations(arbf_inds, o):
            if srbf is None:
                inds = list(inds)
            else:
                inds = list(srbf_inds) + list(inds)
            if len(inds) == 0:
                const += np.sum(alpha)
            elif len(inds) == 1:
                funcps.append(np.dot(alpha, k0s[inds[0]]))
                spline_grids.append(UCGrid(dims[inds[0]]))
                ind_sets.append(tuple(inds))
            elif len(inds) == 2:
                k = np.einsum('ni,nj->nij', k0s[inds[0]], k0s[inds[1]])
                spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]]))
                funcps.append(np.einsum('n,nij->ij', alpha, k))
                ind_sets.append(tuple(inds))
            elif len(inds) == 3:
                fps = 0
                for p0, p1 in pyscf.lib.prange(0, len(k0s[inds[0]]), 200):
                    k = np.einsum(
                        'ni,nj,nk->nijk',
                        k0s[inds[0]][p0:p1],
                        k0s[inds[1]][p0:p1],
                        k0s[inds[2]][p0:p1]
                    )
                    fps += np.einsum('n,nijk->ijk', alpha[p0:p1], k)
                funcps.append(fps)
                spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]],
                                           dims[inds[2]]))
                ind_sets.append(tuple(inds))
            elif len(inds) == 4:
                k = np.einsum('ni,nj,nk,nl->nijkl', k0s[inds[0]], k0s[inds[1]],
                              k0s[inds[2]], k0s[inds[3]])
                funcps.append(np.einsum('n,nijkl->ijkl', alpha, k))
                spline_grids.append(UCGrid(dims[inds[0]], dims[inds[1]],
                                           dims[inds[2]], dims[inds[3]]))
                ind_sets.append(tuple(inds))
            else:
                raise ValueError('Order too high!')

    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    evaluator = NormGPFunctional(scale, ind_sets, spline_grids, coeff_sets,
                                 gpr.xed_y_converter, gpr.feature_list,
                                 gpr.desc_order, const=const,
                                 desc_version=gpr.desc_version,
                                 a0=gpr.a0, fac_mul=gpr.fac_mul,
                                 amin=gpr.amin, args=gpr.args)
    if gpr.args.use_ex_kernel:
        from ciderpress.models.gp import XED_Y_CONVERTERS
        evaluator.xed_y_converter = XED_Y_CONVERTERS['CHACHIYO']

    if n == 1 and order == 2:
        res, en = arbf(X, get_sub_kernels=True)
        resg = srbf(X)
        res = np.dot(alpha, arbf.scale[0] * resg)

        print("Comparing K and Kspline!!!")
        print(en[0])
        print(spline_grids[0][0], coeff_sets[0], ind_sets, X.shape)
        tsp = eval_cubic(spline_grids[0],
                         coeff_sets[0],
                         X[:,1:2])
        diff = (D[:,:1] - D[:,:1].T) / length_scale[0]
        tk = np.exp(-0.5 * diff**2)
        print(np.mean(np.abs(srbf(X) - tk)))
        print(np.mean(np.abs(res - evaluator.predict_from_desc(D, max_order=1))))
        print(np.mean(np.abs(res - arbf.scale[0] * tsp)))
        print(evaluator.scale[0], scale[0], arbf.scale[0])
        print(res[::1000])
        print(tsp[::1000])
        print("checked 1d")
        print(np.mean(res - evaluator.predict_from_desc(D, max_order=1)))
        res += np.dot(alpha, arbf.scale[1] * en[1] * resg)
        print(np.mean(np.abs(res - evaluator.predict_from_desc(D, max_order=2, vec_eval=True)[0])))
        #print(np.mean(np.abs(res - evaluator.predict_from_desc(D, max_order=2))))
        res += np.dot(alpha, arbf.scale[2] * en[2] * resg)
        print(np.mean(np.abs(res - evaluator.predict_from_desc(D, max_order=3, vec_eval=True)[0])))
        #print(np.mean(np.abs(res - evaluator.predict_from_desc(D, max_order=3))))

    ytest = gpr.gp.predict(X)
    ypred = evaluator.predict_from_desc(D)
    print(np.mean(np.abs(ytest - ypred)))
    print(np.mean(np.abs(ytest - y)))
    print(np.mean(np.abs(y - ypred)))
    print(np.linalg.norm(ytest - y))
    print(np.linalg.norm(ypred - y))
    print(ytest.shape)

    if test_x is not None:
        ytest = gpr.xed_to_y(gpr.predict(test_x, test_rho_data), test_rho_data)
        ypred = gpr.xed_to_y(evaluator.predict(test_x, test_rho_data), test_rho_data)
        print(np.max(np.abs(ytest - ypred)))
        print(np.max(np.abs(ytest - test_y)))
        print(np.max(np.abs(test_y - ypred)))
        print()
        print(np.linalg.norm(ytest - test_y))
        print(np.mean(np.abs(ytest - test_y)))
        print(np.mean(ytest - test_y))
        print(np.linalg.norm(ypred - test_y))
        print(np.mean(np.abs(ypred - test_y)))
        print(np.mean(ypred - test_y))
        print(np.linalg.norm(ypred - ytest))
        print(np.mean(np.abs(ypred - ytest)))
        print(np.mean(ypred - ytest))

    return evaluator

def main():
    parser = ArgumentParser()
    parser.add_argument('outname', type=str, help='Spline output filename')
    parser.add_argument('fname', type=str, help='GP filename (model to map)')
    parser.add_argument('-vs', '--validation-set', nargs='+')
    parser.add_argument('--basis', default='def2-qzvppd', type=str,
                        help='basis set code')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('-v', '--version', default='c', type=str,
                        help='version of descriptor set. Default c')
    parser.add_argument('--srbfd', default=8, type=int,
                        help='grid density for reduced gradient descriptor, srbfd pts per stddev of feature')
    parser.add_argument('--arbfd', default=8, type=int,
                        help='grid density for other descriptors, arbfd pts per stddev of feature')
    parser.add_argument('--maxng', default=120, type=int,
                        help='maximum number of grid points for a feature')
    #srbf_density=8, arbf_density=8, max_ngrid=120
    args = parser.parse_args()
    
    print('OUTNAME', args.outname)
    
    gpr = load(args.fname)
    assert (gpr.args.validation_set is None) or len(gpr.args.validation_set) % 2 == 0,\
        'Need pairs of entries for datasets list.'
    nv = 0#len(gpr.args.validation_set) // 2
    if nv != 0:
        Xv, yv, rhov, rho_datav = parse_dataset(gpr.args, 0, val=True)
    for i in range(1, nv):
        Xn, yn, rhon, rho_datan, = parse_dataset(gpr.args, i, val=True)
        Xv = np.append(Xv, Xn, axis=0)
        yv = np.append(yv, yn, axis=0)
        rhov = np.append(rhov, rhon, axis=0)
        rho_datav = np.append(rho_datav, rho_datan, axis=1)
    if not gpr.args.agpr:
        evaluator = get_mapped_gp_evaluator_simple(gpr)
    elif nv == 0:
        evaluator = get_mapped_gp_evaluator(gpr, srbf_density=args.srbfd,
                                            arbf_density=args.arbfd,
                                            max_ngrid=args.maxng)
    else:
        evaluator = get_mapped_gp_evaluator(gpr, test_x=Xv, test_y=yv,
                                            test_rho_data=rho_datav,
                                            srbf_density=args.srbfd,
                                            arbf_density=args.arbfd,
                                            max_ngrid=args.maxng)
    evaluator.fx_baseline = gpr.xed_y_converter[2]
    evaluator.fxb_num = gpr.xed_y_converter[3]
    evaluator.desc_version = gpr.desc_version
    evaluator.amin = gpr.amin
    evaluator.a0 = gpr.a0
    evaluator.fac_mul = gpr.fac_mul
    dump(evaluator, args.outname)

if __name__ == '__main__':
    main()
