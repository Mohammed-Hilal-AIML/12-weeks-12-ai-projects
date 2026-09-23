from pathlib import Path
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as sk_train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split
warnings.filterwarnings('ignore')
BASE=Path(__file__).parent; DATA=BASE/'data'; OUT=BASE/'outputs'; MODELS=BASE/'models'; RNG=42

def load_data():
    rp,mp=DATA/'ratings.csv',DATA/'movies.csv'
    if rp.exists() and mp.exists(): r,m=pd.read_csv(rp),pd.read_csv(mp)
    elif (DATA/'u.data').exists() and (DATA/'u.item').exists():
        r=pd.read_csv(DATA/'u.data',sep='\t',names=['userId','movieId','rating','timestamp'])
        m=pd.read_csv(DATA/'u.item',sep='|',header=None,encoding='latin-1',usecols=[0,1],names=['movieId','title']); m['genres']='Unknown'
    else: raise FileNotFoundError('Put ratings.csv and movies.csv, or u.data and u.item, in data/. See data/README.md.')
    r=r.rename(columns={'user_id':'userId','movie_id':'movieId'}); m=m.rename(columns={'movie_id':'movieId','movie_title':'title'})
    for c in ['userId','movieId']: r[c]=pd.to_numeric(r[c]); m['movieId']=pd.to_numeric(m['movieId'])
    r['rating']=pd.to_numeric(r['rating']); m['title']=m['title'].astype(str); m['genres']=m.get('genres',pd.Series('Unknown',index=m.index)).fillna('Unknown').astype(str)
    return r,m

def content_model(movies):
    x=movies.reset_index(drop=True).copy(); x['content']=x.genres.str.replace('|',' ',regex=False)
    tf=TfidfVectorizer(); mat=tf.fit_transform(x.content); sim=cosine_similarity(mat); idx={t:i for i,t in enumerate(x.title)}
    def similar(title,n=10):
        if title not in idx:
            q=x[x.title.str.contains(title,case=False,na=False)]
            if q.empty: raise ValueError(f'Movie not found: {title}')
            title=q.iloc[0].title
        i=idx[title]; order=np.argsort(sim[i])[::-1]; rows=[]
        for j in order:
            if j!=i: rows.append((x.iloc[j].movieId,x.iloc[j].title,float(sim[i,j])))
            if len(rows)==n: break
        return pd.DataFrame(rows,columns=['movieId','title','content_similarity'])
    return x,sim,idx,similar

def collaborative(user,ratings,movies,n=10):
    mat=ratings.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)
    if user not in mat.index: return pd.DataFrame(columns=['movieId','title','collaborative_score'])
    s=cosine_similarity(mat.loc[[user]],mat)[0]; sims=pd.Series(s,index=mat.index).drop(user).nlargest(5)
    scores=mat.loc[sims.index].T.dot(sims); rated=set(ratings.loc[ratings.userId==user,'movieId']); scores=scores.drop(list(rated),errors='ignore').nlargest(n)
    return pd.DataFrame({'movieId':scores.index,'collaborative_score':scores.values}).merge(movies[['movieId','title']],on='movieId')[['movieId','title','collaborative_score']]

def train_svd(ratings):
    reader=Reader(rating_scale=(float(ratings.rating.min()),float(ratings.rating.max())))
    data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    gs=GridSearchCV(SVD,{'n_factors':[50,100,150],'n_epochs':[20],'lr_all':[.005],'reg_all':[.02]},measures=['rmse'],cv=3,n_jobs=1); gs.fit(data)
    trainset,testset=train_test_split(data,test_size=.2,random_state=RNG); model=SVD(**gs.best_params['rmse'],random_state=RNG); model.fit(trainset)
    preds=model.test(testset); rmse=accuracy.rmse(preds,verbose=False); mae=accuracy.mae(preds,verbose=False)
    with open(MODELS/'svd_model.pkl','wb') as f: pickle.dump(model,f)
    print(f'Best CV RMSE: {gs.best_score["rmse"]:.4f}'); print('Best params:',gs.best_params['rmse']); print(f'Hold-out RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    return model,rmse,mae

def hybrid(user,seed,model,ratings,movies,sim,idx,n=10):
    if seed not in idx:
        q=movies[movies.title.str.contains(seed,case=False,na=False)]
        if q.empty: raise ValueError(f'Seed movie not found: {seed}')
        seed=q.iloc[0].title
    si=idx[seed]; rated=set(ratings.loc[ratings.userId==user,'movieId']); rows=[]
    for _,row in movies.iterrows():
        mid=row.movieId
        if mid in rated: continue
        try: pred=float(model.predict(user,mid).est)
        except Exception: continue
        rows.append((mid,row.title,pred,float(sim[si,idx[row.title]])))
    d=pd.DataFrame(rows,columns=['movieId','title','svd_score','content_similarity'])
    if d.empty:return d.assign(hybrid_score=[])
    lo,hi=d.svd_score.min(),d.svd_score.max(); d['svd_norm']=(d.svd_score-lo)/(hi-lo) if hi>lo else .5
    d['hybrid_score']=.7*d.svd_norm+.3*d.content_similarity
    return d.sort_values('hybrid_score',ascending=False).head(n).drop(columns='svd_norm')

def evaluate(train,test,movies,similar,model,sim,idx,max_users=100):
    pos=test[test.rating>=4].groupby('userId').movieId.apply(set); rows=[]; seen=0
    for u,rel in pos.items():
        if u not in set(train.userId): continue
        ur=train[train.userId==u]
        if ur.empty: continue
        seed=movies[movies.movieId==ur.sort_values('rating',ascending=False).iloc[0].movieId]
        if seed.empty: continue
        title=seed.iloc[0].title
        recs={'Content-Based':set(similar(title,10).movieId),'Collaborative':set(collaborative(u,train,movies,10).movieId),'Hybrid':set(hybrid(u,title,model,train,movies,sim,idx,10).movieId)}
        for method,ids in recs.items():
            hit=len(ids&rel); rows.append((method,hit/10,hit/len(rel)))
        seen+=1
        if seen>=max_users: break
    if not rows:return pd.DataFrame()
    return pd.DataFrame(rows,columns=['method','precision_at_10','recall_at_10']).groupby('method',as_index=False).mean()

def main():
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); ratings,movies=load_data(); merged=ratings.merge(movies,on='movieId',how='left')
    print(f'Ratings: {len(ratings):,} | Users: {ratings.userId.nunique():,} | Movies: {merged.movieId.nunique():,}')
    plt.figure(figsize=(8,5)); ratings.rating.value_counts().sort_index().plot(kind='bar'); plt.title('MovieLens Rating Distribution'); plt.xlabel('Rating'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(OUT/'rating_distribution.png',dpi=150); plt.close()
    ui=ratings.pivot_table(index='userId',columns='movieId',values='rating'); print('User-item matrix:',ui.shape,'| Sparsity:',f'{1-ui.notna().sum().sum()/ui.size:.2%}')
    plt.figure(figsize=(12,7)); plt.imshow(ui.iloc[:30,:40].notna(),aspect='auto',interpolation='nearest'); plt.title('User-Item Matrix Sample'); plt.xlabel('Movies'); plt.ylabel('Users'); plt.tight_layout(); plt.savefig(OUT/'user_item_heatmap.png',dpi=150); plt.close()
    movies,sim,idx,similar=content_model(movies); seed=movies.iloc[0].title; print('\nContent recommendations for:',seed); print(similar(seed,10).to_string(index=False))
    train,test=sk_train_test_split(ratings,test_size=.2,random_state=RNG); train=train.reset_index(drop=True); test=test.reset_index(drop=True); user=int(train.userId.iloc[0]); print('\nCollaborative recommendations for user',user); print(collaborative(user,train,movies,10).to_string(index=False))
    model,rmse,mae=train_svd(train); user_seed=movies[movies.movieId==train[train.userId==user].sort_values('rating',ascending=False).iloc[0].movieId].iloc[0].title; print('\nHybrid recommendations using:',user_seed); print(hybrid(user,user_seed,model,train,movies,sim,idx,10).to_string(index=False))
    metrics=evaluate(train,test,movies,similar,model,sim,idx); print('\nPrecision@10 / Recall@10'); print(metrics.to_string(index=False) if not metrics.empty else 'No eligible users.');
    if not metrics.empty: metrics.to_csv(OUT/'evaluation_summary.csv',index=False)
    pd.DataFrame([{'svd_holdout_rmse':rmse,'svd_holdout_mae':mae,'n_factors':model.n_factors}]).to_csv(OUT/'svd_summary.csv',index=False)
    print('\nDone. Outputs:',OUT)
if __name__=='__main__': main()
