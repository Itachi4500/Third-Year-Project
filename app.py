import os,io,tempfile,warnings
from datetime import datetime
import streamlit as st, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
try: import plotly.express as px
except: px=None
try: import shap
except: shap=None
try: from pycaret.classification import setup as pyc_setup, compare_models, pull as pyc_pull
from pycaret.regression import setup as pyr_setup; HAS_PYCARET=True
except: HAS_PYCARET=False
try: from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
except: pass
try: from fpdf import FPDF; HAS_FPDF=True
except: HAS_FPDF=False
try: import openai; HAS_OPENAI=True
except: HAS_OPENAI=False

st.set_page_config("Enhanced Data Analysis Assistant v3","wide")
for k in ["history","memory","df","models"]: st.session_state.setdefault(k,{} if k=="models" else [])
st.session_state.setdefault("df",None)
def log(a): st.session_state.history.append({"time":datetime.now().isoformat(),"action":a})
def upload(): f=st.file_uploader("CSV/Excel",accept_multiple_files=True); 
if f: dfs=[pd.read_csv(x) if x.name.endswith(".csv") else pd.read_excel(x) for x in f]; return dfs[0] if len(dfs)==1 else pd.concat(dfs,ignore_index=True) if dfs else None
def dtypes(df): return df.dtypes.apply(str).to_dict()
def clean(df): df=df.copy(); df.columns=[c.strip() for c in df.columns]; 
for c in df.columns: 
    if df[c].dtype=="object":
        try: df[c]=pd.to_datetime(df[c])
        except: pass
return df
def pre(df): num=df.select_dtypes(np.number).columns.tolist(); cat=df.select_dtypes(["object","category"]).columns.tolist(); return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer("median")),("scaler",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer("most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",sparse=False))]),cat]),num,cat
def out(df,c,f=1.5): q1,q3=df[c].quantile([0.25,0.75]); iqr=q3-q1; return df[(df[c]<q1-f*iqr)|(df[c]>q3+f*iqr)]
def plot(df): 
    if px: cols=st.multiselect("x,y,color",df.columns.tolist(),default=df.columns[:3].tolist())
    if len(cols)>=2: st.plotly_chart(px.scatter(df,x=cols[0],y=cols[1],color=cols[2] if len(cols)>2 else None),use_container_width=True)
def save(m,n="model.pkl"): joblib.dump(m,n); st.success(f"Saved {n}"); return n
def pdf(title,b,imgs=None,out="report.pdf"):
    if not HAS_FPDF: st.error("FPDF missing"); return
    p=FPDF(); p.set_auto_page_break(True,15); p.add_page(); p.set_font("Arial",16); p.cell(200,10,title,ln=True,align='C'); p.ln(5); p.set_font("Arial",12)
    [p.multi_cell(0,8,x) or p.ln(2) for x in b]
    if imgs: [p.add_page() or tmp:=tempfile.NamedTemporaryFile(delete=False,suffix=".png") or tmp.write(i) or tmp.close() or p.image(tmp.name,10,20,180) for i in imgs]
    p.output(out); return out
def ai(q):
    if not HAS_OPENAI: return "Install openai"
    k=os.getenv("OPENAI_API_KEY")
    if not k: return "Set OPENAI_API_KEY"
    openai.api_key=k
    try: return openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":q}],temperature=0.2).choices[0].message.content
    except Exception as e: return f"OpenAI error: {e}"

st.sidebar.title("🧠 Assistant v3")
mode=st.sidebar.radio("Mode",["Main App","AI Chat","Settings"])
with st.sidebar.expander("Login"): pwd=st.text_input("Password",type="password")
if pwd: st.sidebar.success("Authenticated") if pwd=="prem" else st.sidebar.error("Wrong password")

if mode=="Settings": st.header("Settings"); st.markdown(f"PyCaret: {HAS_PYCARET} | SHAP: {shap is not None} | FPDF: {HAS_FPDF}")
elif mode=="AI Chat": st.header("AI Chat"); q=st.text_area("Ask"); 
if st.button("Ask AI"): st.write(ai(q)); log(f"AI question: {q}")
else:
    st.title("Enhanced Data Analysis Assistant"); t=st.tabs(["Upload","Clean","EDA","Model","Explain","Export","Memory"])
    with t[0]: df=upload(); 
    if df is not None: st.session_state.df=df; st.success("Loaded"); st.dataframe(df.head()); log("Uploaded")
    if st.session_state.df is not None and st.button("Summary"): st.write(st.session_state.df.describe(include='all'))
    with t[1]:
        df=st.session_state.df
        if df is None: st.warning("Upload data")
        else:
            st.write(pd.DataFrame.from_dict(dtypes(df),orient='index',columns=['dtype']))
            if st.button("Quick Clean"): st.session_state.df=clean(df); st.success("Cleaned"); log("Cleaned")
            c=st.selectbox("Outlier col",df.select_dtypes(np.number).columns.tolist())
            if c: o=out(df,c); st.write(f"{len(o)} outliers")
            if st.button("Remove outliers"): st.session_state.df=df[~df.index.isin(o.index)].reset_index(drop=True); st.success("Removed"); log(f"Outliers {c}")
    with t[2]: df=st.session_state.df; st.header("EDA"); 
    if df is None: st.warning("Upload data")
    else: st.write(df.describe(include='all').T); plot(df)
    if st.button("Corr Heatmap") and px: st.plotly_chart(px.imshow(df.select_dtypes(np.number).corr(),text_auto=True),use_container_width=True)
    with t[3]:
        st.header("Model"); df=st.session_state.df
        if df is None: st.warning("Upload data")
        else:
            task=st.selectbox("Task",["Classification","Regression"]); target=st.selectbox("Target",df.columns.tolist()); ts=st.slider("Test size",0.1,0.5,0.2)
            if st.button("Train"):
                X=df.drop(columns=[target]); y=df[target]; X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=ts,random_state=42)
                pre,_ ,_=pre(df.drop(columns=[target]))
                try: pre.fit(X_train); X_train,X_test=pre.transform(X_train),pre.transform(X_test)
                except: X_train,X_test=X_train.fillna(0),X_test.fillna(0)
                m=RandomForestClassifier(50,42) if task=="Classification" else RandomForestRegressor(50,42)
                m.fit(X_train,y_train); p=m.predict(X_test)
                if task=="Classification": st.success(f"Acc: {accuracy_score(y_test,p):.4f}"); st.text(classification_report(y_test,p))
                else: st.success(f"MSE: {mean_squared_error(y_test,p):.4f}")
                n=st.text_input("Model filename","model.pkl")
                if st.button("Save"): save(m,n); st.session_state.models[n]=m; log(f"Saved {n}")
            if HAS_PYCARET and st.button("AutoML"):
                try: s=pyc_setup(df,target=target,silent=True,session_id=123) if task=="Classification" else pyr_setup(df,target=target,silent=True,session_id=123)
                b=compare_models(); st.write(b); st.dataframe(pyc_pull()); log("AutoML")
                except: pass
    with t[4]:
        st.header("Explain")
        if st.session_state.models:
            mn=st.selectbox("Model",list(st.session_state.models.keys())); m=st.session_state.models.get(mn)
            if m and shap: X=df.drop(columns=[st.selectbox("Target",df.columns.tolist())]); shap.plots.beeswarm(shap.Explainer(m,X)(X)); st.pyplot(bbox_inches='tight')
    with t[5]:
        st.header("Export"); df=st.session_state.df
        if df is not None:
            if st.button("CSV"): st.download_button("CSV",df.to_csv(index=False).encode('utf-8'),"data.csv","text/csv")
            title=st.text_input("Report title","Data Analysis Report"); notes=st.text_area("Notes","Add summary")
            if st.button("PDF"):
                imgs=[]
                if px: buf=io.BytesIO(); px.histogram(df.select_dtypes(np.number).iloc[:,0:1].dropna().melt(),x='value').write_image(buf,format='png'); imgs.append(buf.getvalue())
                out=pdf(title,[notes,f"Generated on {datetime.now().isoformat()}"],imgs)
                if out: st.download_button("PDF",open(out,"rb"),"report.pdf"); log("PDF generated")
    with t[6]:
        st.header("Memory"); st.write(pd.DataFrame(st.session_state.history)); st.write(st.session_state.memory)
        with st.form("mem"): k=st.text_input("Key"); v=st.text_area("Value"); s=st.form_submit_button("Save")
        if s and k: st.session_state.memory[k]=v; st.success("Saved"); log(f"Memory {k}")
st.markdown("---"); st.caption("Install optional packages (pycaret, shap, plotly, fpdf) for full features.")
