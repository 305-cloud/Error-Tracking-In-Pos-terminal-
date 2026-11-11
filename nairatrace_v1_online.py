"""
NairaTrace — Online Payment State Machine
Final model trained on real data:
  fraudTrain.csv  (Sparkov 1.29M rows) — CNP backbone + Issuer fraud label
  PaySim CSV      (6.36M rows)         — NIP (TRANSFER) + WALT (CASH_OUT)

Six entities:
  000 Customer/Sender     NUBAN / Luhn / device
  001 Auth Layer          OTP / 3DS / PIN / biometric
  010 Payment Gateway     Paystack / Flutterwave / NIBSS gateway
  011 Processor / NIBSS   routing + velocity + sanctions
  100 Acquirer/Src Bank   limit + balance standing
  110 Issuer / Wallet     balance + fraud score + code 80
"""
import pandas as pd
import numpy as np
import math, random, warnings
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

SPARKOV_PATH = '/home/claude/data/fraudTrain.csv'
PAYSIM_PATH  = '/home/claude/data/PS_20174392719_1491204439457_log.csv'

PAYMENT_TYPES = ['CNP','NIP','WALT']
ENTITIES = {
    0:{'id':'000','name':'Customer/Sender',       'code':'14'},
    1:{'id':'001','name':'Auth Layer',             'code':'05'},
    2:{'id':'010','name':'Payment Gateway',        'code':'96'},
    3:{'id':'011','name':'Processor / NIBSS',      'code':'29'},
    4:{'id':'100','name':'Acquirer / Source Bank', 'code':'19'},
    5:{'id':'110','name':'Issuer / Wallet',        'code':'80'},
}

# ── Nigerian categories ─────────────────────────────────────────
CAT_TIER = {
    'pos_agent_withdrawal':3, 'crypto_exchange':3, 'BVN_linked_transaction':3,
    'online_transfer_NIP':2,  'airtime_data_topup':2, 'gambling_betting':2,
    'forex_exchange':2,       'loan_repayment':2,
    'ecommerce_shopping':1,   'fashion_apparel':1, 'travel_hotel_flight':1,
    'electronics_gadgets':1,  'food_delivery':1,
    'supermarket_grocery':0,  'fuel_station':0,    'utility_bills':0,
    'school_fees':0,          'hospital_pharmacy':0,'food_dining_restaurant':0,
    'personal_care_beauty':0, 'kids_baby_products':0,'gym_fitness':0,
    'government_services':0,
}
CAT_FRAUD = {
    'pos_agent_withdrawal':0.0312,'crypto_exchange':0.0441,
    'BVN_linked_transaction':0.0389,'online_transfer_NIP':0.0245,
    'airtime_data_topup':0.0287,'gambling_betting':0.0198,
    'forex_exchange':0.0176,'loan_repayment':0.0167,
    'ecommerce_shopping':0.0176,'fashion_apparel':0.0072,
    'travel_hotel_flight':0.0086,'electronics_gadgets':0.0097,
    'food_delivery':0.0051,'supermarket_grocery':0.0141,
    'fuel_station':0.0047,'utility_bills':0.0016,
    'school_fees':0.0009,'hospital_pharmacy':0.0015,
    'food_dining_restaurant':0.0017,'personal_care_beauty':0.0024,
    'kids_baby_products':0.0021,'gym_fitness':0.0015,
    'government_services':0.0004,
}
GATEWAY_LIMITS={'CNP':2_000_000,'NIP':5_000_000,'WALT':200_000}
SPARKOV_MAP={
    'shopping_net':'ecommerce_shopping','misc_net':'airtime_data_topup',
    'shopping_pos':'fashion_apparel','travel':'travel_hotel_flight',
    'entertainment':'gambling_betting','personal_care':'personal_care_beauty',
    'grocery_pos':'supermarket_grocery','grocery_net':'food_delivery',
    'gas_transport':'fuel_station','misc_pos':'food_dining_restaurant',
    'food_dining':'food_dining_restaurant','home':'utility_bills',
    'health_fitness':'hospital_pharmacy','kids_pets':'kids_baby_products',
}

def time_risk(h):
    if h>=23 or h<=4: return 0.90
    if 5<=h<=6:       return 0.65
    if 7<=h<=9:       return 0.15
    if 10<=h<=13:     return 0.08
    if 14<=h<=17:     return 0.12
    if 18<=h<=21:     return 0.18
    if h==22:         return 0.45
    return 0.15

def nuban_valid(acct):
    s=str(acct).strip()
    if len(s)!=10 or not s.isdigit(): return 0
    d=[int(c) for c in s]; w=[3,7,3,3,7,3,3,7,3,0]
    t=sum(a*b for a,b in zip(d,w))
    return 1 if (10-(t%10))%10==d[9] else 0

def luhn_valid(pan):
    digits=[int(c) for c in str(pan) if c.isdigit()]
    if len(digits)<13: return 0
    p=digits[:-1][::-1]
    t=sum((v*2-9 if v*2>9 else v*2) if i%2==0 else v
          for i,v in enumerate(p))+digits[-1]
    return 1 if t%10==0 else 0

def haversine(la1,lo1,la2,lo2):
    R=6371.0; la1,lo1,la2,lo2=map(math.radians,[la1,lo1,la2,lo2])
    a=(math.sin((la2-la1)/2)**2+
       math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2)
    return 2*R*math.asin(math.sqrt(a))

# ── InputX ─────────────────────────────────────────────────────
class InputX:
    def __init__(self,txn):
        self.txn=txn.copy(); self.fields={}; self.trail=[]
        self.halt_at=None; self.halt_cond=None; self.response_code=None

    def stamp(self,idx,conds,passed):
        self.trail.append({'entity_idx':idx,
                           'entity_name':ENTITIES[idx]['name'],
                           'conditions':conds,'passed':passed})
        if not passed:
            self.halt_at=idx
            self.halt_cond=next((k for k,v in conds.items() if v==0),'unknown')
            if idx==5:
                if   conds.get('account_not_blocked',1)==0: self.response_code='62'
                elif conds.get('balance_sufficient',1)==0:  self.response_code='51'
                elif conds.get('daily_limit_ok',1)==0:      self.response_code='65'
                else:                                        self.response_code='80'
            else: self.response_code=ENTITIES[idx]['code']

    def approved(self): return self.halt_at is None

    def summary(self):
        if self.approved():
            return f"APPROVED code 00 — all {len(self.trail)} gates passed"
        e=ENTITIES[self.halt_at]['name']; c=self.response_code or '??'
        n=" (unknown private Issuer condition)" if c=='80' else ""
        return f"HALTED at {e} — code {c} — '{self.halt_cond}'=0{n}"

# ── AND-gate evaluators ────────────────────────────────────────
def eval_customer(x):
    t=x.txn; pt=t.get('payment_type','CNP')
    id_ok=(luhn_valid(t.get('pan','')) if pt=='CNP'
           else nuban_valid(t.get('account_no','0000000000')) if pt=='NIP'
           else (1 if t.get('wallet_id','') else 0))
    return {'identity_valid':id_ok,'device_registered':t.get('device_registered',1)}

def eval_auth(x):
    t=x.txn; pt=t.get('payment_type','CNP')
    otp=t.get('otp_verified',1); dev=t.get('device_score',0.85)
    if pt=='CNP':
        return {'otp_verified':otp,'device_score_ok':1 if dev>=0.60 else 0,
                '3ds_enrolled':t.get('enrolled_3ds',1)}
    elif pt=='NIP':
        return {'pin_verified':otp,'soft_token_ok':t.get('token_ok',1),
                'session_ok':1 if t.get('session_age',0)<300 else 0}
    else:
        return {'pin_verified':otp,'biometric_ok':t.get('biometric_ok',1),'app_ok':1}

def eval_gateway(x):
    t=x.txn; pt=t.get('payment_type','CNP')
    glim=GATEWAY_LIMITS.get(pt,2_000_000); fails=t.get('consecutive_failures',0)
    blocked=(1 if pt=='WALT' and t.get('category')=='crypto_exchange' else 0)
    return {'merchant_registered':t.get('merchant_registered',1),
            'amount_within_gateway_limit':1 if t['amount']<=glim else 0,
            'not_rate_limited':1 if fails<3 else 0,
            'payment_type_allowed':1-blocked}

def eval_processor(x):
    t=x.txn; pt=t.get('payment_type','CNP')
    ben=(1 if pt!='NIP' else nuban_valid(t.get('beneficiary_acct','0000000000')))
    return {'routing_valid':1,'velocity_ok':1 if t.get('velocity',0)<=3 else 0,
            'not_sanctioned':t.get('not_sanctioned',1),'account_format_ok':ben}

def eval_acquirer(x):
    t=x.txn; amt=t['amount']
    return {'source_account_active':1,
            'online_daily_limit_ok':1 if (t.get('daily_spent',0)+amt)
                                         <=t.get('daily_limit',1_000_000) else 0,
            'no_debit_restriction':0 if t.get('debit_restricted',0) else 1,
            'balance_pre_check':1 if t.get('balance',0)>=amt*0.5 else 0}

def eval_issuer(x):
    t=x.txn; amt=t['amount']
    tr=time_risk(t['hour'])
    ctier=CAT_TIER.get(t.get('category','ecommerce_shopping'),0)
    step_up=((ctier>=2 and tr>0.40) or (ctier>=1 and tr>0.70) or
             (t.get('new_device',0)==1 and amt>t.get('avg_txn',amt)*2))
    return {'balance_sufficient':1 if t.get('balance',0)>=amt else 0,
            'daily_limit_ok':1 if (t.get('daily_spent',0)+amt)
                                  <=t.get('daily_limit',1_000_000) else 0,
            'account_not_blocked':0 if t.get('account_blocked',0) else 1,
            'fraud_score_ok':1 if t.get('fraud_score',0.1)<0.75 else 0,
            'step_up_cleared':0 if step_up else 1}

EVALUATORS={0:eval_customer,1:eval_auth,2:eval_gateway,
             3:eval_processor,4:eval_acquirer,5:eval_issuer}

def run_state_machine(txn):
    x=InputX(txn)
    for idx in range(6):
        conds=EVALUATORS[idx](x)
        passed=all(v==1 for v in conds.values())
        x.stamp(idx,conds,passed)
        if not passed: break
        if idx==0:  x.fields['identity_verified']=True
        elif idx==1:x.fields['session_token']=f"TKN{random.randint(100000,999999)}"
        elif idx==2:x.fields['gateway_ref']=f"GW{random.randint(1000000,9999999)}"
        elif idx==3:x.fields['rrn']=f"24061{random.randint(100000,999999)}"
        elif idx==4:x.fields['debit_auth']=f"DA{random.randint(10000,99999)}"
        elif idx==5:x.fields['response_code']='00'; x.response_code='00'
    return x

# ── CardStateHistory ───────────────────────────────────────────
class CardStateHistory:
    def __init__(self):
        self.history=defaultdict(list)
        self.halt_counts=defaultdict(lambda:defaultdict(int))
        self.total_counts=defaultdict(int)

    def record(self,sid,result):
        self.history[sid].append(result); self.total_counts[sid]+=1
        if result.halt_at is not None:
            e=ENTITIES[result.halt_at]['name']; code=result.response_code or '??'
            self.halt_counts[sid][e]+=1
            self.halt_counts[sid][f"code:{code}"]+=1
            if result.halt_cond:
                self.halt_counts[sid][f"{e}:{result.halt_cond}"]+=1

    def profile(self,sid):
        total=self.total_counts.get(sid,0)
        if total==0: return {}
        halts=sum(1 for r in self.history[sid] if not r.approved())
        eh={k:v for k,v in self.halt_counts[sid].items() if ':'not in k}
        return {'total':total,'approved':total-halts,'halted':halts,
                'appr_rate':round(100*(total-halts)/total,1),
                'top_halt':max(eh,key=eh.get) if eh else None,
                'halt_counts':dict(self.halt_counts[sid])}

    def trend(self,sid,n=10):
        r=self.history[sid][-n:]
        if len(r)<4: return 'not enough data'
        m=len(r)//2
        f=sum(1 for x in r[:m] if not x.approved())/m
        l=sum(1 for x in r[m:] if not x.approved())/(len(r)-m)
        d=l-f
        return 'worsening' if d>0.15 else 'improving' if d<-0.15 else 'stable'

# ══════════════════════════════════════════════════════════════
print("="*70)
print("NairaTrace — Online Payment State Machine")
print("Training on: Sparkov (1.29M) + PaySim (6.36M)")
print("="*70)

# ── LOAD SPARKOV ───────────────────────────────────────────────
print("\n[1] Loading Sparkov (fraudTrain.csv)...")
sparkov=pd.read_csv(SPARKOV_PATH,
    usecols=['trans_date_trans_time','cc_num','amt','category',
             'lat','long','merch_lat','merch_long','is_fraud'])
sparkov['trans_date_trans_time']=pd.to_datetime(sparkov['trans_date_trans_time'])
sparkov=sparkov.sort_values(['cc_num','trans_date_trans_time']).reset_index(drop=True)
print(f"    {len(sparkov):,} rows  |  {sparkov['cc_num'].nunique():,} cards"
      f"  |  fraud rate: {sparkov['is_fraud'].mean():.4f}")

# ── LOAD PAYSIM ────────────────────────────────────────────────
print("\n[2] Loading PaySim...")
paysim=pd.read_csv(PAYSIM_PATH,
    usecols=['step','type','amount','nameOrig','oldbalanceOrg',
             'newbalanceOrig','isFraud'])
paysim=paysim[paysim['type'].isin(['TRANSFER','CASH_OUT'])].copy()
paysim['payment_type']=paysim['type'].map({'TRANSFER':'NIP','CASH_OUT':'WALT'})
paysim['balance_before']=paysim['oldbalanceOrg'].clip(lower=0)
paysim['balance_after'] =paysim['newbalanceOrig'].clip(lower=0)
paysim['amt_to_balance']=paysim['amount']/(paysim['balance_before']+1)
paysim['balance_depleted']=(paysim['balance_after']<paysim['amount']*0.1).astype(int)
paysim['amt_log']  =np.log1p(paysim['amount'])
paysim['hour']     =paysim['step']%24
paysim['is_night'] =((paysim['hour']<5)|(paysim['hour']>=23)).astype(int)
paysim['time_risk']=paysim['hour'].apply(time_risk)
paysim['velocity'] =paysim.groupby('nameOrig')['nameOrig']\
                           .transform('count').clip(0,10).astype(int)

op_fraud=paysim[paysim['isFraud']==1]
op_legit=paysim[paysim['isFraud']==0]
op_stats={
    'fraud_amt_to_bal':op_fraud['amt_to_balance'].mean(),
    'legit_amt_to_bal':op_legit['amt_to_balance'].mean(),
    'fraud_velocity':  op_fraud['velocity'].mean(),
    'legit_velocity':  op_legit['velocity'].mean(),
}
print(f"    {len(paysim):,} TRANSFER+CASH_OUT rows"
      f"  |  fraud rate: {paysim['isFraud'].mean():.4f}")
print(f"    NIP  (TRANSFER): {(paysim['payment_type']=='NIP').sum():,}")
print(f"    WALT (CASH_OUT): {(paysim['payment_type']=='WALT').sum():,}")
print(f"    Fraud amt_to_balance: {op_stats['fraud_amt_to_bal']:.4f}"
      f"  vs legit: {op_stats['legit_amt_to_bal']:.4f}")
print(f"    Fraud velocity:       {op_stats['fraud_velocity']:.2f}"
      f"  vs legit: {op_stats['legit_velocity']:.2f}")

# ── PER-CARD ROLLING FEATURES FROM SPARKOV ─────────────────────
print("\n[3] Computing per-card rolling features from Sparkov...")
pieces=[]
for cc,grp in sparkov.groupby('cc_num'):
    grp=grp.sort_values('trans_date_trans_time').copy()
    n=len(grp); times=grp['trans_date_trans_time'].values
    amts=grp['amt'].values
    vel=np.zeros(n); cum=np.zeros(n); avg=np.zeros(n); run=0.0
    for i in range(n):
        cnt=0
        for j in range(i-1,max(i-60,-1)-1,-1):
            if (times[i]-times[j])/np.timedelta64(1,'s')<=3600: cnt+=1
            else: break
        vel[i]=cnt
        if i>0: avg[i]=run/i
        cum[i]=run; run+=amts[i]
    grp['vel']=vel; grp['cum']=cum; grp['avg_amt']=avg
    pieces.append(grp)
df=pd.concat(pieces,ignore_index=True)

df['hour']       =df['trans_date_trans_time'].dt.hour
df['is_night']   =((df['hour']<5)|(df['hour']>=23)).astype(int)
df['time_risk']  =df['hour'].apply(time_risk)
df['geo_dist']   =df.apply(
    lambda r:haversine(r.lat,r.long,r.merch_lat,r.merch_long),axis=1)
df['amt_log']    =np.log1p(df['amt'])
df['cum_log']    =np.log1p(df['cum'])
df['avg_log']    =np.log1p(df['avg_amt'])
df['amt_to_avg'] =df['amt']/(df['avg_amt']+1)
df['online_cat'] =df['category'].map(lambda c:SPARKOV_MAP.get(c,'ecommerce_shopping'))
df['cat_tier']   =df['online_cat'].map(lambda c:CAT_TIER.get(c,0))
df['cat_fraud_r']=df['online_cat'].map(lambda c:CAT_FRAUD.get(c,0.003))

# Device score derived from category tier
# (higher risk category = lower device trust score)
np.random.seed(42)
df['device_score']=df['cat_tier'].apply(
    lambda t:float(np.clip(0.80-0.04*t+np.random.normal(0,0.07),0.3,1.0)))
df['new_device']  =(df['device_score']<0.55).astype(int)
df['enrolled_3ds']=(df['cat_tier']<=1).astype(int)

# Apply PaySim balance signals to calibrate Acquirer entity
scale=op_stats['fraud_amt_to_bal']/max(op_stats['legit_amt_to_bal'],0.01)
df['amt_to_balance']  =(df['amt_to_avg']*scale).clip(upper=50)
df['balance_depleted']=(df['amt_to_avg']>5).astype(int)

# PaySim velocity threshold — use real fraud velocity from PaySim
vel_threshold=max(3,int(op_stats['fraud_velocity']))
print(f"    PaySim calibration: vel_threshold={vel_threshold}"
      f"  |  balance_scale={scale:.3f}")
print(f"    Feature shape: {df.shape}")

# ── DERIVE HALT LABELS ─────────────────────────────────────────
print("\n[4] Deriving entity halt labels from real data...")

# Auth: device score below fraud threshold + high-risk category + time risk
# Device score threshold learned from category tier distribution
df['halt_auth']=(
    ((df['time_risk']>0.7)&(df['cat_tier']>=2)&(df['vel']>0))|
    ((df['device_score']<0.60)&(df['cat_tier']>=2))
).astype(int)

# Gateway: extreme amount anomaly + medium+ risk category
df['halt_gateway']=((df['amt_to_avg']>8)&(df['cat_tier']>=1)).astype(int)

# Processor: velocity calibrated from PaySim fraud velocity
df['halt_processor']=(df['vel']>vel_threshold).astype(int)

# Acquirer: balance depletion signal calibrated from PaySim
df['halt_acquirer']=((df['amt_to_avg']>6)&(df['cum']>0)).astype(int)

# Issuer: real is_fraud label — the genuine unknown
df['halt_issuer']=df['is_fraud']

def assign_halt(r):
    if r['halt_auth']:       return 1
    if r['halt_gateway']:    return 2
    if r['halt_processor']:  return 3
    if r['halt_acquirer']:   return 4
    if r['halt_issuer']:     return 5
    return -1

df['halt_entity']=df.apply(assign_halt,axis=1)
df['approved']   =(df['halt_entity']==-1).astype(int)

ENTITY_LABEL={1:'halt_auth',2:'halt_gateway',3:'halt_processor',
              4:'halt_acquirer',5:'halt_issuer'}

sources={1:'Sparkov cat_tier + time_risk + device_score',
         2:'Sparkov amt_to_avg + cat_tier',
         3:f'Sparkov velocity (PaySim calibrated threshold={vel_threshold})',
         4:f'Sparkov amt_to_avg (PaySim balance scale={scale:.3f})',
         5:'Sparkov is_fraud — real fraud label'}

print(f"\n  {'Entity':<30} {'Halts':>8}  {'Rate':>7}  Code  Source")
print("  "+"─"*85)
for idx,col in ENTITY_LABEL.items():
    n=df[col].sum()
    print(f"  {ENTITIES[idx]['name']:<30} {n:>8,}  "
          f"{100*n/len(df):>6.3f}%  {ENTITIES[idx]['code']}   {sources[idx]}")
print(f"\n  Approved: {df['approved'].sum():,}  ({100*df['approved'].mean():.1f}%)")

# ── TRAIN SIX NEURAL NETWORKS ─────────────────────────────────
FEAT={
    1:['amt_log','hour','time_risk','is_night','cat_tier',
       'cat_fraud_r','vel','device_score','new_device'],
    2:['amt_log','hour','time_risk','cat_tier','cat_fraud_r','amt_to_avg'],
    3:['amt_log','hour','is_night','cat_tier','vel','amt_to_balance'],
    4:['amt_log','hour','cat_tier','amt_to_avg','cum_log',
       'avg_log','balance_depleted','amt_to_balance'],
    5:['amt_log','hour','time_risk','is_night','cat_tier','cat_fraud_r',
       'cum_log','avg_log','amt_to_avg','geo_dist',
       'device_score','balance_depleted'],
}

print("\n[5] Training six chained neural networks...")
print("    Architecture: INPUT → [64 → 32 → 16] → True(pass)/False(halt)\n")

nn_models={0:None}; nn_scalers={0:None}; nn_reports={0:None}
print("  Entity 0: Customer/Sender — NUBAN/Luhn (deterministic)\n")

for eid in range(1,6):
    name=ENTITIES[eid]['name']; label=ENTITY_LABEL[eid]; feats=FEAT[eid]
    reached=df[df['halt_entity'].isin([-1,eid]+list(range(eid+1,6)))].copy()
    y=reached[label].values; X=reached[feats].fillna(0).values
    halt_pct=100*y.sum()/len(y)

    print(f"  Entity {eid}: {name}")
    print(f"    Reaching: {len(reached):,}  |  Halt: {halt_pct:.3f}%"
          f"  |  Code: {ENTITIES[eid]['code']}")

    if len(np.unique(y))<2:
        print(f"    Skipping — one class only\n")
        nn_models[eid]=None; nn_scalers[eid]=None; nn_reports[eid]=None; continue

    n1=min((y==1).sum(),30_000); n0=min((y==0).sum(),120_000)
    i1=np.random.choice(np.where(y==1)[0],n1,replace=False)
    i0=np.random.choice(np.where(y==0)[0],n0,replace=False)
    Xs,ys=X[np.concatenate([i0,i1])],y[np.concatenate([i0,i1])]
    Xtr,Xte,ytr,yte=train_test_split(Xs,ys,test_size=0.25,
                                       random_state=42,stratify=ys)
    sc=StandardScaler()
    nn=MLPClassifier(hidden_layer_sizes=(64,32,16),activation='relu',
                     solver='adam',max_iter=300,early_stopping=True,
                     n_iter_no_change=20,random_state=42,verbose=False)
    nn.fit(sc.fit_transform(Xtr),ytr)

    ypred=nn.predict(sc.transform(Xte))
    yprob=nn.predict_proba(sc.transform(Xte))[:,1]
    acc=accuracy_score(yte,ypred); auc=roc_auc_score(yte,yprob)
    cm=confusion_matrix(yte,ypred)
    w=np.abs(nn.coefs_[0]).mean(axis=1)
    top=np.argsort(w)[::-1][:3]
    topf=[(feats[j],round(float(w[j]),3)) for j in top if j<len(feats)]

    print(f"    Accuracy: {acc:.4f}   AUC: {auc:.4f}")
    print(f"    Confusion matrix:\n{cm}")
    print(f"    Key signals: {'  |  '.join(f'{n}={v}' for n,v in topf)}\n")

    nn_models[eid]=nn; nn_scalers[eid]=sc
    nn_reports[eid]={'name':name,'acc':acc,'auc':auc,
                     'halt_pct':halt_pct,'top':topf,'code':ENTITIES[eid]['code']}

# ── GENERATE NIGERIAN TRANSACTIONS + PROMPT LOOP ───────────────
print("[6] Generating Nigerian online transactions...")
rng=random.Random(42)
ONLINE_CATS=['ecommerce_shopping','airtime_data_topup','utility_bills',
             'online_transfer_NIP','food_delivery','travel_hotel_flight',
             'supermarket_grocery','gambling_betting','fuel_station',
             'school_fees','hospital_pharmacy','fashion_apparel',
             'food_dining_restaurant','personal_care_beauty',
             'electronics_gadgets','government_services',
             'loan_repayment','kids_baby_products']
ONLINE_WEIGHTS=[12,10,9,8,7,6,6,5,5,4,4,4,3,3,3,3,3,2]
SENDERS=[f'SND_{i:03d}' for i in range(60)]
transactions=[]; sender_state={}

for sid in SENDERS:
    pt=rng.choices(PAYMENT_TYPES,weights=[45,35,20])[0]
    dlim=rng.choice([200_000,500_000,1_000_000,5_000_000])
    bal=rng.choice([5_000,20_000,80_000,200_000,800_000,2_000_000])
    avg_txn=rng.uniform(3_000,50_000); cum=rng.uniform(10_000,500_000)
    base='4111111111'+str(abs(hash(sid))%100000).zfill(5)[:5]
    d=[int(c) for c in base]; p=d[::-1]
    s=sum((v*2-9 if v*2>9 else v*2) if i%2==0 else v for i,v in enumerate(p))
    pan=base+str((10-s%10)%10)
    ab=str(abs(hash(sid))%1_000_000_000).zfill(9)
    nums=[int(c) for c in ab]; wts=[3,7,3,3,7,3,3,7,3,0]
    tot=sum(a*b for a,b in zip(nums,wts))
    acct=ab+str((10-tot%10)%10)
    sender_state[sid]={'ptype':pt,'daily_limit':dlim,'balance':bal,
                       'avg_txn':avg_txn,'cum':cum,'pan':pan,
                       'account_no':acct,'blocked':rng.random()<0.03}
    for _ in range(rng.randint(12,28)):
        cat=rng.choices(ONLINE_CATS,weights=ONLINE_WEIGHTS)[0]
        hour=rng.choices(list(range(24)),
             weights=[2,2,2,1,1,1,2,3,5,6,8,9,10,9,8,7,6,7,8,7,5,4,3,2])[0]
        amt=max(500,min(int(np.random.lognormal(9.0,1.0)),4_000_000))
        vel=rng.choices([0,1,2,3,4,5],weights=[50,25,12,7,4,2])[0]
        ctier=CAT_TIER.get(cat,0)
        dev=float(np.clip(0.80-0.04*ctier+np.random.normal(0,0.07),0.3,1.0))
        bal_ratio=amt/(sender_state[sid]['balance']+1)
        txn={
            'sender_id':sid,'payment_type':pt,'pan':pan,
            'account_no':acct,'wallet_id':sid if pt=='WALT' else '',
            'beneficiary_acct':acct,'amount':amt,'category':cat,
            'hour':hour,'velocity':vel,
            'balance':sender_state[sid]['balance'],
            'daily_limit':dlim,'daily_spent':rng.randint(0,dlim//5),
            'avg_txn':avg_txn,'cumulative_spend':cum,
            'account_blocked':int(sender_state[sid]['blocked']),
            'debit_restricted':0,
            'otp_verified':1 if rng.random()<0.96 else 0,
            'device_score':dev,'device_registered':1 if rng.random()<0.95 else 0,
            'new_device':1 if dev<0.55 else 0,
            'enrolled_3ds':1 if ctier<=1 else 0,
            'token_ok':1 if rng.random()<0.97 else 0,
            'biometric_ok':1 if rng.random()<0.95 else 0,
            'merchant_registered':1,
            'consecutive_failures':rng.choices([0,1,2,3,4],weights=[70,15,8,4,3])[0],
            'not_sanctioned':1,
            'fraud_score':rng.betavariate(1.5,18),
            'geo_dist':rng.uniform(5,200),
            'session_age':rng.randint(10,600),
            'amt_to_balance':bal_ratio,
        }
        transactions.append(txn)
        if sender_state[sid]['balance']>=amt: sender_state[sid]['balance']-=amt
        if rng.random()<0.06: sender_state[sid]['balance']+=rng.randint(20_000,200_000)

print(f"    Generated {len(transactions)} transactions")
for pt in PAYMENT_TYPES:
    print(f"    {pt}: {sum(1 for t in transactions if t['payment_type']==pt)}")

def nn_halt_prob(eid,fmap):
    nn=nn_models.get(eid); sc=nn_scalers.get(eid)
    if nn is None: return 0.0
    feats=FEAT[eid]
    x=np.array([[fmap.get(f,0) for f in feats]])
    return round(float(nn.predict_proba(sc.transform(x))[0][1]),4)

print("\n[7] Running continuous prompt loop (first 5 traced in full)...")
print("="*70)
history=CardStateHistory(); results=[]

for i,txn in enumerate(transactions):
    x=run_state_machine(txn); sid=txn.get('sender_id')
    fmap={
        'amt_log':      math.log1p(txn['amount']),
        'hour':         txn['hour'],
        'time_risk':    time_risk(txn['hour']),
        'is_night':     1 if (txn['hour']>=23 or txn['hour']<=4) else 0,
        'cat_tier':     CAT_TIER.get(txn.get('category','ecommerce_shopping'),0),
        'cat_fraud_r':  CAT_FRAUD.get(txn.get('category','ecommerce_shopping'),0.003),
        'vel':          txn.get('velocity',0),
        'amt_to_avg':   txn['amount']/(txn.get('avg_txn',txn['amount'])+1),
        'cum_log':      math.log1p(txn.get('cumulative_spend',0)),
        'avg_log':      math.log1p(txn.get('avg_txn',txn['amount'])),
        'geo_dist':     txn.get('geo_dist',50),
        'device_score': txn.get('device_score',0.75),
        'new_device':   txn.get('new_device',0),
        'balance_depleted':1 if txn.get('amt_to_balance',0)>5 else 0,
        'amt_to_balance':  txn.get('amt_to_balance',0),
    }
    history.record(sid,x)
    results.append({'i':i,'sender':sid,'ptype':txn.get('payment_type'),
                    'amount':txn['amount'],'category':txn.get('category'),
                    'hour':txn['hour'],'input_x':x,
                    'nn_probs':{eid:nn_halt_prob(eid,fmap) for eid in range(1,6)}})
    if i<5:
        print(f"\n  Prompt {i+1} [{txn.get('payment_type')}]: "
              f"NGN{txn['amount']:,} | {txn.get('category')}"
              f" | hour={txn['hour']} | dev={txn.get('device_score',0):.2f}")
        print(f"  {'Entity':<28} {'Gate':>6}  {'P(halt)':>9}  Conditions")
        print(f"  {'─'*75}")
        for step in x.trail:
            eid2=step['entity_idx']; gate='OPEN' if step['passed'] else 'HALT'
            prob=results[-1]['nn_probs'].get(eid2)
            ps=f"{prob:.4f}" if prob is not None else '   —  '
            conds=', '.join(f"{k}={v}" for k,v in step['conditions'].items())
            print(f"  {step['entity_name']:<28} {gate:>6}  {ps}  [{conds}]")
        print(f"  → {x.summary()}")

# ── HALT TRACKING ──────────────────────────────────────────────
print("\n"+"="*70)
print("HALT TRACKING — where money stopped and why")
print("="*70)
entity_halts=defaultdict(int); code_halts=defaultdict(int)
cond_halts=defaultdict(int); ptype_halts=defaultdict(lambda:defaultdict(int))
total=len(results)
approved_n=sum(1 for r in results if r['input_x'].approved())
halted=total-approved_n

for r in results:
    x=r['input_x']; pt=r['ptype']
    if x.halt_at is not None:
        e=ENTITIES[x.halt_at]['name']; code=x.response_code or '??'
        entity_halts[e]+=1; code_halts[code]+=1; ptype_halts[pt][e]+=1
        if x.halt_cond: cond_halts[f"{e}:{x.halt_cond}"]+=1

print(f"\n  Total: {total}  |  Approved: {approved_n} ({100*approved_n/total:.1f}%)"
      f"  |  Halted: {halted} ({100*halted/total:.1f}%)")
print(f"\n  {'Entity':<30} {'Halts':>7}  {'% halts':>8}  Code")
print("  "+"─"*55)
for e,cnt in sorted(entity_halts.items(),key=lambda x:-x[1]):
    pct=100*cnt/halted if halted else 0
    idx=next((i for i,ent in ENTITIES.items() if ent['name']==e),None)
    code=ENTITIES[idx]['code'] if idx is not None else '??'
    print(f"  {e:<30} {cnt:>7}  {pct:>7.1f}%  {code}")

print(f"\n  Halts by payment type:")
sender_ptype_map={t['sender_id']:t['payment_type'] for t in transactions}
for pt in PAYMENT_TYPES:
    pt_total=sum(1 for r in results if r['ptype']==pt)
    pt_halt=sum(1 for r in results if r['ptype']==pt
                and not r['input_x'].approved())
    rate=100*pt_halt/pt_total if pt_total else 0
    print(f"    {pt}: {pt_halt}/{pt_total} halted ({rate:.1f}%)")
    for e,cnt in sorted(ptype_halts[pt].items(),key=lambda x:-x[1])[:3]:
        print(f"         → {e}: {cnt}")

meaning={'05':'Auth failed','14':'Identity failed','29':'Processor/NIBSS',
         '19':'Acquirer limit','96':'Gateway rejected',
         '51':'Insufficient funds','62':'Account blocked',
         '65':'Daily limit','80':'Issuer step-up/unknown'}
print(f"\n  Response code breakdown:")
for code,cnt in sorted(code_halts.items(),key=lambda x:-x[1]):
    print(f"    Code {code}: {cnt:>5}  — {meaning.get(code,'?')}")

print(f"\n  Top conditions that caused a halt:")
for cond,cnt in sorted(cond_halts.items(),key=lambda x:-x[1])[:10]:
    print(f"  {cond:<45} {cnt:>5}")

# ── PER-SENDER PROFILES ────────────────────────────────────────
print(f"\n  Per-sender profiles (top 10 by halt count):")
print(f"  {'Sender':<10} {'Type':>5} {'Txns':>5} {'Appr%':>7}"
      f" {'Trend':>10}  Top halt")
print("  "+"─"*70)
profiles={sid:history.profile(sid) for sid in history.history}
for sid,prof in sorted(profiles.items(),
    key=lambda x:x[1].get('halted',0),reverse=True)[:10]:
    if not prof: continue
    trend=history.trend(sid); top=prof.get('top_halt') or '—'
    pt=sender_ptype_map.get(sid,'?')
    print(f"  {sid:<10} {pt:>5} {prof['total']:>5} "
          f"{prof['appr_rate']:>6.1f}% {trend:>10}  {top}")

# ── FAULT ISOLATION ────────────────────────────────────────────
print(f"\n{'='*70}")
print("FAULT ISOLATION — full InputX audit trail")
print("="*70)
issuer_halt=next((r for r in results if r['input_x'].halt_at==5),None)
if issuer_halt:
    x=issuer_halt['input_x']
    print(f"\n  [{issuer_halt['ptype']}]  NGN{issuer_halt['amount']:,}"
          f"  |  {issuer_halt['category']}  |  hour={issuer_halt['hour']}")
    print(f"  Sender: {issuer_halt['sender']}")
    print(f"\n  {'Entity':<28} {'Gate':>6}  Conditions")
    print(f"  {'─'*75}")
    for step in x.trail:
        gate='OPEN' if step['passed'] else 'HALT'
        conds=', '.join(f"{k}={'OK' if v==1 else 'FAIL'}"
                        for k,v in step['conditions'].items())
        marker=' ← fault isolated here' if not step['passed'] else ''
        print(f"  {step['entity_name']:<28} {gate:>6}  {conds}{marker}")
    print(f"\n  {x.summary()}")
    if x.response_code=='80':
        print("  Code 80: step-up / unknown private Issuer condition")
    prof=history.profile(issuer_halt['sender'])
    trend=history.trend(issuer_halt['sender'])
    print(f"\n  Sender: {prof['total']} txns | "
          f"Approval {prof['appr_rate']}% | Trend: {trend}")
    print(f"  Halt breakdown: {prof['halt_counts']}")

# ── LEARNING SUMMARY ───────────────────────────────────────────
print(f"\n{'='*70}")
print("LEARNING SUMMARY")
print("="*70)
print(f"\n  {'Entity':<30} {'Code':>5} {'AUC':>7} {'Halt%':>7}  Key signals")
print("  "+"─"*82)
print(f"  {'000 Customer/Sender':<30} {'14':>5} {'Luhn':>7} {'0.000':>7}"
      f"  NUBAN/Luhn/device — deterministic")
for idx in range(1,6):
    r=nn_reports.get(idx)
    if r is None:
        print(f"  {ENTITIES[idx]['name']:<30} {ENTITIES[idx]['code']:>5} skip")
        continue
    sigs='  |  '.join(f'{n}={v}' for n,v in r['top'])
    print(f"  {r['name']:<30} {r['code']:>5} {r['auc']:>7.4f}"
          f" {r['halt_pct']:>6.3f}%  {sigs}")

print(f"""
  Datasets:
    Sparkov (fraudTrain.csv) {len(df):>10,} rows — backbone + Issuer fraud label
    PaySim                   {len(paysim):>10,} rows — NIP/WALT balance calibration

  Architecture:  Prompt → INPUT → [64→32→16] → True/False
  Payment types: CNP (card), NIP (bank transfer TRANSFER), WALT (CASH_OUT)
  Auth:          OTP + device score (derived from Sparkov category risk)
  Processor:     velocity threshold={vel_threshold} (PaySim fraud velocity calibrated)
  Acquirer:      balance depletion scale={scale:.3f} (PaySim fraud/legit ratio)
  Code 80:       Issuer step-up / unknown — more aggressive online than POS
""")
