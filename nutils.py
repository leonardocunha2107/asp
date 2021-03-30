import numpy as np
from sklearn.decomposition import NMF


def sdp_test(cx):
    u=np.random.randn(cx.shape[-1])+1j*np.random.randn(cx.shape[-1])
    us=np.conj(u[None,None,None,:])
    u=u[None,None,:,None]
    print('sdp ',((us@cx@u).real<0).sum())
    
def T(x):
    x=x.conjugate()
    if len(x.shape)>=3:
        lx=len(x.shape)
        perm=tuple(range(lx-2))+(lx-1,lx-2)
        return x.transpose(*perm)
    
    else: return x.T
def covx(A,W,H,cov_b,part,return_covs=False):    
    """
    X: FxNxI
    A: FxIxJ
    W: FxK
    H: KxN
    cov_b: FxIxI
    OBS: Does not
    """
    
    F,I,J=A.shape
    K,N=H.shape
    #E-Step
    
    ##compute auxiliary quantitities
    
    pscov_c=T(W[:,:,None]*H[None,:,:])[:,:,:,None] #FxNxKx1

    arrs=[pscov_c[:,:,idxs].sum(axis=2).squeeze() for idxs in part]
    
    pscov_s=np.stack(arrs,axis=-1) [:,:,:,None] #FxNxJx1

    # Computing cov_x  A_f is IxJ S_,sfn is psuedo JxJ

    if return_covs:
        return pscov_s.squeeze()
    
    A=A[:,None,:,:] #Fx1xIxJ
    
    #sdp_test(A@(pscov_s*T(A)))
    cov_x=A@(pscov_s*T(A))+cov_b[:,None,:,:] #if len(cov_b.shape)==3 else cov_b #FxNxIxI   
    #cov_b[:,np.arange(I),np.arange(I)]=0
    #print('cb',np.linalg.norm(cov_b))

    assert cov_x.shape==(F,N,I,I)
    return cov_x

def generate_params(I,J,K,F,N,part,cb_diagonal=False,cb_distr=(0.01,0.01)):
    A=np.random.rand(F,I,J)+1j*np.random.rand(F,I,J)
    W=np.random.rand(F,K)#+1j*np.random.rand(F,K)
    H=np.random.rand(K,N)#+1j*np.random.rand(K,N)
    #A,W,H=normalize_parameters(A,W,H,part)
    print(A.shape)
    if cb_diagonal:
        cb=np.stack([np.diag(cb_distr[0]+cb_distr[1]*np.random.rand(I)) for _ in range(F)])
    else:  ##TODO: correct this
        cb=0.05*np.random.randn(I)
        cb=cb[:,None]*cb[None,:]
    
    return A,W,H,cb

def generate_data(I,J,K,F,N,part,cb_diagonal=True,cb_distr=(0.01,0.01)):
    
    A,W,H,cb=generate_params(I,J,K,F,N,part,cb_diagonal,cb_distr=cb_distr)
    
    sigs=covx(A,W,H,cb,part,True)  ##FxNxJ
    s=np.zeros((F,N,J),dtype=np.complex)
    noise=np.stack([np.random.multivariate_normal(np.zeros(I),t,size=(N)) for t in cb])
    for f in range(F):
        for n in range(N):
            s[f,n,:]=np.random.multivariate_normal(np.zeros(J),np.diag(sigs[f,n,]))+1j* \
            np.random.multivariate_normal(np.zeros(J),np.diag(sigs[f,n,]))
    
    X=np.squeeze(A[:,None,:,:]@s[:,:,:,None])+noise
    
    return X,s,A,W,H,cb


def normalize_parameters(A,W,H,part):
    EPS=1e-7
    F,I,J=A.shape
    
    #norm_a=np.linalg.norm(A,axis=1) ##FxJ
    #norm_a*=np.sign(A[:,0,:])
    norm_a=np.sqrt(np.sum(np.real(A*np.conj(A)),axis=1))
    Anew=A/norm_a[:,None,:]*np.exp(-1j*np.angle(A[:,0,:]))[:,None,:] # a_{0,j,f} \in R^+
    
    Wnew=np.zeros_like(W)
    Hnew=np.zeros_like(H)
    for j in range(J):
        Wnew[:,part[j]]=W[:,part[j]]*(norm_a[:,j]**2)[:,None]
        norm_j=np.sum(Wnew[:,part[j]],axis=0)
        Wnew[:,part[j]]=Wnew[:,part[j]]/(norm_j[None,:]+EPS)
        Hnew[part[j],:]=H[part[j],:]*norm_j[:,None]
    
    #norm_w=np.linalg.norm(w,axis=0)
    #W=W/norm_w[None,:]
    
    if(np.isnan(Wnew).sum()):
        print('na',norm_a)
        print(W.sum(axis=0))
        print(A[:,0,:])
        raise Error
    
    
    #H=norm_W[:,None]*H
                     
    return Anew,Wnew,Hnew
def perturbate(srcs,coef_add_noise = 1e-2, coef_mult_noise = 1e-3, coef_mix = .05):
    srcs=srcs.T
    matrix = coef_mix * np.ones((srcs.shape[0], srcs.shape[0])) + np.diag((1 - srcs.shape[0] * coef_mix) * np.ones(srcs.shape[0]))
    perturbated_srcs = np.multiply(srcs, np.random.normal(1, coef_mult_noise, srcs.shape)) + np.random.normal(0, coef_add_noise, srcs.shape)
    return perturbated_srcs.T

def test_normalize(I,J,K,F,N,part):
    X,s,A,W,H,cb=generate_data(I,J,K,F,N,part)
    An,Wn,Hn=normalize_parameters(A,W,H,part)


    print((np.linalg.norm(covx(A,W,H,cb,part)-covx(An,Wn,Hn,cb,part))))

def squared_module(arr):
    return np.multiply(arr, arr.conjugate()).real
def is_div(x,y):
    t=x/y
    return (t-np.log(t)-1).mean()

def is_nmf(strue,part,niter=500,nmf_noise=None):
    F,N,J=strue.shape
    K=sum([len(p) for p in part])
    W,H=np.ones((F,K)),np.ones((K,N))
    for i,p in enumerate(part):

        nmf=NMF(len(p),beta_loss='itakura-saito',solver='mu')
        si=squared_module(strue[:,:,i].T)
        H[p,:]=nmf.fit_transform(si).T
        W[:,p]=nmf.components_.T
    if nmf_noise:
        W=W+np.random.uniform(0,nmf_noise,W.shape)
        H=H+np.random.uniform(0,nmf_noise,H.shape)
    
    return W,H
    
def em_assertions():
    assert pscov_c.shape==(F,N,K) 
    assert pscov_s.shape==(F,N,K)  
    assert cov_x.shape==(F,N,I,I) 
    assert shat.shape==(F,N,J) 
    assert chat.shape==(F,N,K)
    
    assert Rxx.shape==(I,I,F)
    assert Rxs.shape==(I,J,F)
    assert Rss.shape==(J,J,F)
    assert u.shape==(K,F,N)
    
    assert Anew.shape==A.shape
    assert Wnew.shape==W.shape
    assert Hnew.shape==H.shape