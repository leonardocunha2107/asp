import numpy as np

def covx(A,W,H,cov_b,part,return_covs=False):
    """
    X: IxFxN
    A: IxJxF
    W: FxK
    H: KxN
    cov_b: IxI
    """
    
    I,J,F=A.shape
    K,N=H.shape
    
    #E-Step
    
    ##compute auxiliary quantitities

    pscov_c=(W[:,:,None]*H[None,:,:]).transpose(1,0,2) ##KxFxN
    pscov_s=np.stack([pscov_c[idxs,:,:].sum(axis=0) for idxs in part])  ##JxFXN
    
 
    
    # Computing cov_x  A_f is IxJ S_,sfn is psuedo JxJ
    Ah=A.transpose((1,0,2))
    cov_x=pscov_s[:,None,:,:]*Ah[:,:,:,None]    ##JxIxFxN
    cov_x=(A[:,:,None,:,None]*cov_x[None,:,:,:,:]).sum(axis=1) ##IxIxFxN
    
    if return_covs:
        return cov_x ,pscov_s
    return cov_x

def normalize_parameters(A,W,H,part):
    
    I,J,F=A.shape
    
    norm_a=np.linalg.norm(A,axis=0) ##JxF
    Anew=A/norm_a[None,:,:] #
    
    Wnew=np.zeros_like(W)
    for j in range(J):
        W[:,part[j]]=W[:,part[j]]*(norm_a[j]**2)[:,None]
    
    W=W/(W.sum(axis=0)[None,:])
    for j in range(J):
        norm_wj=np.linalg.norm(W[:,part[j]],axis=0)
        
        H[part[j],:]=H[part[j],:]*norm_wj[:,None]
    #W=W*((norm_a**2)[:,None])
    
    #norm_w=np.linalg.norm(w,axis=0)
    #W=W/norm_w[None,:]
    
    #H=norm_W[:,None]*H
                     
    return A,W,H

def test_normalize(I,J,K,F,N,part):
    X,s,A,W,H,cb=generate_data(I,J,K,F,N,part)
    An,Wn,Hn=normalize_parameters(A,W,H,part)

    
    #print(np.linalg.norm(covx(A,W,H,cb,part)-covx(An,Wn,Hn,cb,part)))
                     
def generate_params(I,J,K,F,N,part,cb_diagonal=False):
    A=np.random.randn(I,J,F)
    W=np.random.rand(F,K)
    H=np.random.rand(K,N)
    
    if cb_diagonal:
        cb=np.diag(0.01+0.1*np.random.rand(I))
    else:
        cb=0.05*np.random.randn(I)
        cb=cb[:,None]*cb[None,:]
    
    return A,W,H,cb
    
def generate_data(I,J,K,F,N,part,cb_diagonal=False):
    
    A,W,H,cb=generate_params(I,J,K,F,N,part,cb_diagonal)
    
    
    _,sigs=covx(A,W,H,cb,part,True)
    s=np.zeros((J,F,N))
    noise=np.random.multivariate_normal(np.zeros(I),cb,size=(F,N)).transpose(2,0,1)
    for f in range(F):
        for n in range(N):
            s[:,f,n]=np.random.multivariate_normal(np.zeros(J),np.diag(sigs[:,f,n]))
    
    X=(A[:,:,:,None]*s[None,:,:,:]).sum(axis=1)+noise
    
    return X,s,A,W,H,cb



def sdr(s,shat):
    
    ##JxFxN
    res=(s**2).sum(axis=(1,2))/((s-shat)**2).sum(axis=(1,2))
    
    return 10*np.log10(res)
    

def em_assertions():
    assert pscov_c.shape==(K,F,N) 
    assert pscov_s.shape==(J,F,N)  
    assert cov_x.shape==(I,I,F,N) 
    assert shat.shape==(J,F,N) 
    assert chat.shape==(K,F,N)
    
    assert Rxx.shape==(I,I,F)
    assert Rxs.shape==(I,J,F)
    assert Rss.shape==(J,J,F)
    assert u.shape==(K,F,N)
    
    assert Anew.shape==A.shape
    assert Wnew.shape==W.shape
    assert Hnew.shape==H.shape
    
def old_em_iter(X,A,W,H,cov_b,part):
    """
    X: IxFxN
    A: IxJxF
    W: FxK
    H: KxN
    cov_b: IxI
    OBS: Does not
    """
    
    I,J,F=A.shape
    K,N=H.shape
    
    #E-Step
    
    ##compute auxiliary quantitities

    pscov_c=(W[:,:,None]*H[None,:,:]).transpose(1,0,2) ##KxFxN
    pscov_s=np.stack([pscov_c[idxs,:,:].sum(axis=0) for idxs in part])  ##JxFXN

    #print(pscov_c[,(J,F,N))
    
    # Computing cov_x  A_f is IxJ S_,sfn is psuedo JxJ
    Ah=A.transpose((1,0,2))
    
    cov_x=pscov_s[:,None,:,:]*Ah[:,:,:,None]    ##JxIxFxB
    cov_x=(A[:,:,None,:,None]*cov_x[None,:,:,:,:]).sum(axis=1) ##IxIxFxN
    #cov_x=np.tensordot(A,cov_x,axes=(1,0))
    
    
    Ahat=np.zeros((I,K,F))
    for j,idxs in enumerate(part):
        Ahat[:,idxs,:]=A[:,j,:][:,None,:]

    sigxx=np.linalg.solve(cov_x.transpose(2,3,0,1),X.transpose(1,2,0)).transpose(2,0,1)  ##IxFxN
    

    chat=(pscov_c[:,None,:,:]*(Ahat[:,:,:,None].transpose(1,0,2,3)))*sigxx[None,:,:,:] ## KxIxFxN then KxFxN
    chat=chat.sum(axis=1)
        
    shat=(pscov_s[:,None,:,:]*(A[:,:,:,None].transpose(1,0,2,3)))*sigxx[None,:,:,:]
    shat=shat=shat.sum(axis=1)   ##JxFxN
    
    ##compute sufficient statistics
    
    Rxx=(X[:,None,:,:]*X[None,:,:,:]).mean(axis=-1) ##IxIxF
    Rxs=(X[:,None,:,:]*shat[None,:,:,:]).mean(axis=-1)  ##IxJxF
    Rss=(shat[:,None,:,:]*shat[None,:,:,:]).mean(axis=-1) ##JxJxF
    
    Gc=(pscov_c[:,None,:,:]*(Ahat[:,:,:,None].transpose(1,0,2,3)))[:,None,:,:,:]
    Gc=(Gc*np.linalg.inv(cov_x.transpose(2,3,0,1)).transpose(2,3,0,1)).sum(axis=1)  #KxIxFxN
    
    assert Gc.shape==(K,I,F,N)
    
    u=-(Gc[:,:,None,:,:]*(Ahat[None,:,:,:,None]) ).sum(axis=1)
    u=u*pscov_c[None,:,:,:]
    u+=chat[:,None,:,:]*chat[None,:,:,:]
    u=u[np.arange(K),np.arange(K),:,:]+pscov_c ##KxFxN
    
    ##M step
    
    
    
    Anew=(Rxs[:,:,None,:]*np.linalg.inv(Rss.transpose(2,0,1)).transpose(2,1,0)[None,:,:,:]).sum(axis=1)
    Wnew=(u/H[:,None,:]).mean(axis=-1).T
    Hnew=(u/W.T[:,:,None]).mean(axis=1)
    
    cb=np.zeros((I,I,F))
    ara=((A[:,:,None,:]*Rss[None,:,:,:]).sum(axis=1)[:,:,None,:]*Ah[None,:,:,:]).sum(axis=1)
    ars=(A[:,:,None,:]*(Rxs.transpose(1,0,2))[None,:,:]).sum(axis=1)
    cb[np.arange(I),np.arange(I),:]=(Rxx-ars-ars.transpose(1,0,2)+ara)[np.arange(I),np.arange(I),:]
    
    
    
    ##normalization
    
    Anew,Wnew,Hnew=utils.normalize_parameters(Anew,Wnew,Hnew,part)
    
    return shat,Anew,Wnew,Hnew,cb
    