from nutils import T
import nutils
import numpy as np
from tqdm import tqdm
def    positive(x):
    return (x<0).sum()==0

def crit1(A,W,H,X,cb,part):

    cx=nutils.covx(A,W,H,cb,part)
    #nutils.sdp_test(cx)
    det=np.linalg.det(cx)
    reg=np.log(det).mean()

    lh=np.trace(X[:,:,:,None]@X[:,:,None,:]@np.linalg.inv(cx),axis1=-2,axis2=-1).mean()
    return reg,lh

def sdr(s,shat):
    
    #FxNxJ
    res=(np.absolute(s)**2).sum(axis=(0,1))/(np.absolute(s-shat)**2).sum(axis=(0,1))
    
    return 10*np.log10(res)
                       
def run(n_iter,X,A,W,H,part,cb=None,covb_callable=None,true_s=None,cb_fix=True,isotropic=False):
    assert (cb is not None or covb_callable) and not (cb is not None and covb_callable)
    
    An,Wn,Hn=A,W,H
    
    if true_s is not None: error=np.zeros((n_iter,A.shape[2]))
    crit=[]
    
    for i in tqdm(range(n_iter)):
        if covb_callable is None:
            shat,An,Wn,Hn,cbnew=em_iter(X,An,Wn,Hn,cb,part,isotropic=isotropic)           
            #print("hey")

            if not cb_fix: 
                cb=cbnew

        else:
            if type(covb_callable) not in [np.ndarray,float,np.array]:
                cb=np.stack([np.diag(X.shape[-1]*[covb_callable(i)]) for _ in range(F)])
            shat,An,Wn,Hn,cb=em_iter(X,An,Wn,Hn,cb,part)

        if true_s is not None: error[i]=sdr(true_s,shat)
        assert positive(Wn) and positive(Hn)
        #print('cb',cb.mean())
        #print('norms ',np.linalg.norm(Wn),np.log10(np.linalg.norm(Hn)),      np.log10(np.linalg.norm(cb)))
        #print('isreal',np.imag(W),np.imag(H))
        
        crit.append(crit1(An,Wn,Hn,X,cb,part))


    return shat,An,Wn,Hn,cb,(error,[t[0] for t in crit],[t[1] for t in crit])

def hvar(x):
    return (x*np.conj(x)).real.mean(axis=(1,2))/100

def em_iter(X,A,W,H,cov_b,part,isotropic=False):
    EPS=1e-7
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
    Ahat=np.zeros((F,I,K),dtype=complex)
    for j,idxs in enumerate(part):
        Ahat[:,:,idxs]=A[:,:,j][:,:,None]
    
    Ahat=Ahat[:,None,:,:] ##Fx1xIxK
    A=A[:,None,:,:] #Fx1xIxJ
    X=X[:,:,:,None]
    
    cov_x=A@(pscov_s*T(A))+cov_b[:,None,:,:] #if len(cov_b.shape)==3 else cov_b #FxNxIxI
     
    if(np.isnan(cov_x).sum()):
        print('W',W)
        print('cs',H)
        raise Error
     
    
    assert cov_x.shape==(F,N,I,I)
    
    Gc=pscov_c*(T(Ahat)@np.linalg.inv(cov_x)) ##F,N,K,I
    Gs=pscov_s*(T(A)@np.linalg.inv(cov_x))## FxNxJxI
    
    chat,shat=Gc@X,Gs@X ##FxNxI FxNxJ
    assert chat.shape==(F,N,K,1)
    assert shat.shape==(F,N,J,1)
  
    Rxx=(X@T(X)).mean(axis=1) #FxIxI
    Rxs=(X@T(shat)).mean(axis=1)  ##FxIxJ
    
    Rss=(shat@T(shat))
    Rss-=(Gs@A)*T(pscov_s) 
    Rss[:,:,np.arange(J),np.arange(J)]+=pscov_s.squeeze()
    Rss=Rss.mean(axis=1)
    #FxIxI

    u=(((chat@T(chat))-(Gc@Ahat)*T(pscov_c)))[:,:,np.arange(K),np.arange(K)].real ##JxJxF
    u+=pscov_c.squeeze() ##F,N,K
    assert u.shape==(F,N,K)
    
    assert H.shape==(K,N)
    
    ##M step
    #assert positive(u)
    
    
    Anew=Rxs@np.linalg.inv(Rss)
    Wnew=(T(u)/(H[None,:,:]+EPS)).mean(axis=-1)
    Hnew=(T(u)/(W[:,:,None]+EPS)).mean(axis=0)
    

    
    A=A.squeeze()
    cbnew=Rxx-A@T(Rxs)-Rxs@T(A)+A@Rss@T(A)
    
    if isotropic:
        cbnew=np.stack([np.eye(I)*np.trace(t)/I for t in cbnew])
   
    else:
        cbnew=np.stack([np.diag(np.diag(t)) for t in cbnew])
    
    cbnew=cbnew.real
   
    
    
    ##normalization
    assert all([a.shape==b.shape for a,b in zip((A,W,H,cov_b),(Anew,Wnew,Hnew,cbnew))])
    
    Anew,Wnew,Hnew=nutils.normalize_parameters(Anew,Wnew,Hnew,part)
        
        
    return shat.squeeze(),Anew,Wnew,Hnew,cbnew