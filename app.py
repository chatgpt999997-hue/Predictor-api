from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import math, numpy as np, pandas as pd
from xgboost import XGBClassifier

app = FastAPI(title="Predictor API")

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_dn = dn.ewm(com=n-1, adjust=False).mean()
    rs = ma_up/(ma_dn+1e-12)
    return 100-(100/(1+rs))

def yf_feats(sym, lookback=180, interval="1d"):
    import yfinance as yf
    df = yf.download(sym, period=f"{lookback+60}d", interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.title)[["Open","High","Low","Close","Volume"]].dropna().copy()
    df["ret1"]=df["Close"].pct_change()
    df["ret5"]=df["Close"].pct_change(5)
    df["ret10"]=df["Close"].pct_change(10)
    df["vol10"]=df["Close"].rolling(10).std()
    df["sma5"]=df["Close"].rolling(5).mean()
    df["sma20"]=df["Close"].rolling(20).mean()
    df["sma50"]=df["Close"].rolling(50).mean()
    df["sma5_div_20"]=df["sma5"]/df["sma20"]
    df["sma20_div_50"]=df["sma20"]/df["sma50"]
    df["rsi14"]=rsi(df["Close"],14)
    return df.dropna().copy()

def temporal_binary_last(feats, label_col):
    cols=[c for c in feats.columns if c not in ["Open","High","Low","Close","Volume",label_col]]
    X=feats[cols].values; y=feats[label_col].values
    n=len(feats); ntr=max(int(n*0.8),50)
    if len(np.unique(y[:ntr]))<2:
        y[:ntr] = np.where(np.arange(ntr)%2==0, 0, 1)
    model=XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=2, reg_lambda=1.0
    )
    model.fit(X[:ntr], y[:ntr])
    return float(model.predict_proba(X[-1:])[:,1][0])

def to_confidence(p):
    p = min(max(float(p),1e-6), 1-1e-6)
    return float(1.0/(1.0+math.exp(-8.0*(p-0.5))))

class PredictReq(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    horizon: int = 1
    lookback: int = 180

@app.post("/predict")
def predict(body: PredictReq):
    out = []
    now = datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
    for sym in [s.strip() for s in body.symbols if s.strip()]:
        feats = yf_feats(sym, body.lookback, body.timeframe)
        if feats is None or feats.empty or len(feats)<60:
            out.append({"symbol": sym, "timestamp": now, "error": "NO_DATA"})
            continue
        fwd = feats["Close"].shift(-body.horizon)
        feats["y_up"]=(fwd>feats["Close"]).astype(int)
        feats = feats.iloc[:-body.horizon].dropna()
        p = temporal_binary_last(feats, "y_up")
        direction = "long" if p>=0.5 else "short"
        cur = float(feats["Close"].iloc[-1])
        vol = float(feats["vol10"].iloc[-1]) if "vol10" in feats else 0.0
        tgt = cur + (vol if direction=="long" else -vol)
        out.append({
            "symbol": sym,
            "timestamp": now,
            "prediction": {
                "direction": direction,
                "target": round(tgt,4),
                "horizon": f"{body.horizon}{body.timeframe}",
                "current": round(cur,4)
            },
            "confidence": round(to_confidence(p),6),
            "risk": {"vol": round(vol,6)},
            "features": {}
        })
    return out
