import numpy as np
def T(x):
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
    
    pscov_c=(W[:,:,None]*H[None,:,:]).transpose(0,2,1)[:,:,:,None] #FxNxKx1

    arrs=[pscov_c[:,:,idxs].sum(axis=2).squeeze() for idxs in part]
    
    pscov_s=np.stack(arrs,axis=-1) [:,:,:,None] #FxNxJx1

    # Computing cov_x  A_f is IxJ S_,sfn is psuedo JxJ

    if return_covs:
        return pscov_s.squeeze()
    
    A=A[:,None,:,:] #Fx1xIxJ
    
    
    
    cov_x=A@(pscov_s*T(A))+cov_b[:,None,:,:] #if len(cov_b.shape)==3 else cov_b #FxNxIxI
    
    return cov_x

def generate_params(I,J,K,F,N,part,cb_diagonal=False,cb_distr=(0.01,0.01)):
    A=np.random.rand(F,I,J)
    W=np.random.rand(F,K)
    H=np.random.rand(K,N)
    A,W,H=normalize_parameters(A,W,H,part)
    
    if cb_diagonal:
        cb=np.stack([np.diag(cb_distr[0]+cb_distr[1]*np.random.rand(I)) for _ in range(F)])
    else:  ##TODO: correct this
        cb=0.05*np.random.randn(I)
        cb=cb[:,None]*cb[None,:]
    
    return A,W,H,cb

def generate_data(I,J,K,F,N,part,cb_diagonal=True):
    
    A,W,H,cb=generate_params(I,J,K,F,N,part,cb_diagonal)
    
    sigs=covx(A,W,H,cb,part,True)  ##FxNxJ
    s=np.zeros((F,N,J))
    noise=np.stack([np.random.multivariate_normal(np.zeros(I),t,size=(N)) for t in cb])
    for f in range(F):
        for n in range(N):
            s[f,n,:]=np.random.multivariate_normal(np.zeros(J),np.diag(sigs[f,n,]))
    
    X=np.squeeze(A[:,None,:,:]@s[:,:,:,None])+noise
    
    return X,s,A,W,H,cb


def normalize_parameters(A,W,H,part):
    
    F,I,J=A.shape
    
    norm_a=np.linalg.norm(A,axis=1) ##FxJ
    norm_a*=np.sign(A[:,0,:])
    Anew=A/norm_a[:,None,:] #
    
    Wnew=np.zeros_like(W)
    Hnew=np.zeros_like(H)
    for j in range(J):
        Wnew[:,part[j]]=W[:,part[j]]*(norm_a[:,j]**2)[:,None]
        norm_j=np.sum(Wnew[:,part[j]],axis=0)
        Wnew[:,part[j]]=Wnew[:,part[j]]/norm_j[None,:]
        Hnew[part[j],:]=H[part[j],:]*norm_j[:,None]
    
    #norm_w=np.linalg.norm(w,axis=0)
    #W=W/norm_w[None,:]
    
    #H=norm_W[:,None]*H
                     
    return Anew,Wnew,Hnew

def test_normalize(I,J,K,F,N,part):
    X,s,A,W,H,cb=generate_data(I,J,K,F,N,part)
    An,Wn,Hn=normalize_parameters(A,W,H,part)


    print((np.linalg.norm(covx(A,W,H,cb,part)-covx(An,Wn,Hn,cb,part))))
    
def is_div(x,y):
    t=x/y
    return (t-np.log(t)-1).mean()

def is_nmf(s,K,niter=500):
    F,N=s.shape
    W,H=np.ones((F,K)),np.ones((K,N))
    error=[]
    for i in tqdm(range(niter)):
        WH=W@H
        error.append(is_div(s,WH))
        
        Hn=H*(W.T@((WH**-2)*s))/(W.T@(WH**-1))
        Wn=W*((((WH**-2)*s)@H.T)/
              ((WH**-1)@H.T))
        ##normalize
        wnorm=np.linalg.norm(Wn,axis=0)
        W=Wn/wnorm[None,:]
        H=Hn*wnorm[:,None]
    
    return W,H,error
    
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