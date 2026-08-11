"""
Workforce Intelligence Hub — PDO (synthetic data)
Corporate PowerBI/PowerApps style, green PDO theme.
Pages: Executive Summary | Talent Profile | Dream Team
ALL DATA SYNTHETIC.
"""
import os, json, re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Workforce Intelligence Hub",
                   page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

DATA_FILE = "pdo_talent_profiles.csv"
MODEL = "claude-sonnet-4-5-20250929"

GREEN="#00843D"; GREEN_DARK="#005A2B"; GREEN_DEEP="#0A3D2C"
LIME="#7AC143"; GOLD="#F2A900"; RED="#E24A33"; SLATE="#5D6D7E"
GREEN_SCHEME=["#0A3D2C","#005A2B","#00843D","#4CAF50","#7AC143","#B4E197"]

# ---- HTML helper: strip indentation so Streamlit never shows raw <div> ----
def H(s): return "\n".join(l.strip() for l in s.strip().split("\n"))
def html(s): st.markdown(H(s), unsafe_allow_html=True)

# =========================================================
# CSS
# =========================================================
html(f"""
<style>
#MainMenu, footer {{visibility:hidden;}}
/* keep header area but make it transparent so the sidebar expand arrow stays visible */
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
/* ALWAYS show the control that re-opens a collapsed sidebar */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{
  visibility:visible !important; opacity:1 !important; display:flex !important;
  z-index:1000000 !important; }}
[data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button {{
  background:{GREEN} !important; color:#fff !important; border-radius:8px !important; }}
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg {{
  color:#fff !important; fill:#fff !important; }}
.stApp {{ background:#EDF2EE; }}
.block-container {{ padding-top:0.6rem; padding-bottom:2rem; max-width:1500px; }}
.wih-header {{ background:linear-gradient(100deg,{GREEN_DEEP} 0%,{GREEN_DARK} 45%,{GREEN} 100%);
 border-radius:16px; padding:16px 26px; margin-bottom:16px; display:flex; align-items:center;
 gap:16px; box-shadow:0 6px 22px rgba(0,90,43,.28); }}
.wih-title {{ color:#fff; font-size:2rem; font-weight:800; margin:0; }}
.wih-sub {{ color:#CDEBD6; font-size:.92rem; margin-top:2px; }}
.wih-badge {{ margin-left:auto; background:rgba(255,255,255,.14); color:#fff; padding:8px 16px;
 border-radius:30px; font-size:.82rem; font-weight:600; border:1px solid rgba(255,255,255,.25);}}
.kpi {{ background:#fff; border-radius:14px; padding:14px 16px; min-height:112px;
 box-shadow:0 3px 12px rgba(16,40,30,.08); border-top:4px solid {GREEN}; }}
.kpi .lbl {{ color:{SLATE}; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.4px;}}
.kpi .val {{ color:{GREEN_DEEP}; font-size:1.75rem; font-weight:800; line-height:1.15; margin-top:4px;}}
.kpi .sub {{ color:#8A97A6; font-size:.76rem; }}
.kpi.alt {{ border-top-color:{GOLD}; }} .kpi.alt2 {{ border-top-color:{LIME}; }} .kpi.alt3 {{ border-top-color:{SLATE}; }}
.sec {{ font-size:1.2rem; font-weight:800; color:{GREEN_DEEP}; margin:10px 0 6px 0;
 border-left:5px solid {GREEN}; padding-left:12px; }}
.card {{ background:#fff; border-radius:14px; padding:16px 18px; margin-bottom:12px;
 box-shadow:0 3px 12px rgba(16,40,30,.07); }}
.pill {{ display:inline-block; padding:3px 11px; border-radius:20px; font-size:.76rem; font-weight:700; margin:2px 3px;}}
.pill.g {{ background:#E3F3E8; color:{GREEN_DARK}; }} .pill.y {{ background:#FDF0D5; color:#9A6B00; }}
.pill.r {{ background:#FBE3DE; color:#A93226; }} .pill.s {{ background:#EBEFF3; color:{SLATE}; }}
.rowitem {{ background:#fff; border-radius:12px; padding:12px 15px; margin-bottom:8px;
 box-shadow:0 2px 8px rgba(16,40,30,.06); border-left:4px solid {GREEN}; }}
.bar {{ background:#E6ECE8; border-radius:6px; height:8px; margin-top:6px; }}
.bar > div {{ height:8px; border-radius:6px; }}
section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{GREEN_DEEP},{GREEN_DARK}); min-width:290px !important; }}
section[data-testid="stSidebar"] * {{ color:#EAF6EE; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ background:rgba(255,255,255,.06);
 border:1px solid rgba(255,255,255,.10); border-radius:12px; padding:14px 16px; margin-bottom:10px;
 width:100%; font-weight:600; cursor:pointer; transition:.15s;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:rgba(255,255,255,.16); transform:translateX(3px);}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background:#fff; border-color:#fff;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:{GREEN_DARK} !important; font-weight:800;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:1.05rem !important; }}
.sb-brand {{ text-align:center; padding:4px 0 14px 0; border-bottom:1px solid rgba(255,255,255,.15); margin-bottom:16px;}}
.sb-brand h3 {{ color:#fff; margin:6px 0 0 0; font-size:1.12rem; font-weight:800;}}
.sb-brand span {{ color:#BFE6CC; font-size:.76rem; }}
div[data-testid="stMetric"] {{ background:#fff; border-radius:12px; padding:12px 14px;
 box-shadow:0 3px 12px rgba(16,40,30,.07); border-top:4px solid {GREEN};}}
.stButton>button {{ background:{GREEN}; color:#fff; border:none; border-radius:10px; padding:11px 22px; font-weight:700;}}
.stButton>button:hover {{ background:{GREEN_DARK}; color:#fff; }}
/* bigger search inputs */
.stTextArea textarea {{ font-size:1.05rem; border:2px solid {GREEN}; border-radius:12px;
 background:#F8FFFB; min-height:90px; }}
.stTextArea textarea:focus {{ border-color:{LIME}; box-shadow:0 0 0 3px rgba(122,193,67,.25);}}
/* team structure */
.team-lead {{ background:linear-gradient(135deg,{GREEN},{GREEN_DARK}); color:#fff; border-radius:16px;
 padding:16px 22px; text-align:center; width:300px; margin:0 auto 6px auto; box-shadow:0 6px 18px rgba(0,90,43,.3);}}
.team-lead .n {{ font-size:1.15rem; font-weight:800; }}
.team-lead .r {{ font-size:.82rem; color:#CDEBD6; }}
.connector {{ width:2px; height:22px; background:{GREEN}; margin:0 auto; opacity:.5;}}
.member {{ background:#fff; border-radius:12px; padding:11px 13px; box-shadow:0 2px 8px rgba(16,40,30,.08);
 border-top:3px solid {LIME}; height:100%;}}
.member .n {{ font-weight:800; color:{GREEN_DEEP}; font-size:.95rem;}}
.member .t {{ color:{SLATE}; font-size:.76rem; }}
.member .m {{ font-size:.76rem; margin-top:4px;}}
.gcap {{ text-align:center; color:{SLATE}; font-size:.82rem; font-weight:700; margin-top:-8px;}}
</style>
""")

LOGO=f"""<svg width="52" height="52" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="2" y="2" width="60" height="60" rx="16" fill="white" fill-opacity="0.12" stroke="white" stroke-opacity="0.5"/>
<circle cx="32" cy="22" r="7" fill="{LIME}"/><circle cx="19" cy="40" r="6" fill="#fff"/>
<circle cx="45" cy="40" r="6" fill="#fff"/><path d="M32 29c-6 0-9 6-9 6M32 29c6 0 9 6 9 6M25 44h14"
stroke="{LIME}" stroke-width="3" stroke-linecap="round"/></svg>"""

def header():
    html(f"""<div class="wih-header">{LOGO}<div>
    <p class="wih-title">Workforce Intelligence Hub</p>
    <div class="wih-sub">Petroleum Development Oman · Talent & Succession Analytics</div>
    </div><div class="wih-badge">🌿 11,000 profiles · synthetic</div></div>""")

def kpi(col,label,value,sub="",cls=""):
    col.markdown(H(f"""<div class="kpi {cls}"><div class="lbl">{label}</div>
    <div class="val">{value}</div><div class="sub">{sub}</div></div>"""),unsafe_allow_html=True)

def sec(t): html(f'<div class="sec">{t}</div>')

def barrow(col_or_st,label,val,color,right=""):
    col_or_st.markdown(H(f"""<div class="rowitem" style="border-left-color:{color}">
    <b>{label}</b> <span style="float:right;font-weight:800;color:{color}">{right}</span>
    <div class="bar"><div style="background:{color};width:{max(3,min(100,val))}%"></div></div></div>"""),
    unsafe_allow_html=True)

def gauge(value,good_high=True):
    v=float(value)
    if good_high: color=GREEN if v>=75 else (GOLD if v>=55 else RED)
    else:         color=RED if v>=66 else (GOLD if v>=40 else GREEN)
    src=pd.DataFrame({"k":["v","rest"],"val":[v,100-v]})
    arc=alt.Chart(src).mark_arc(innerRadius=54,outerRadius=76,cornerRadius=4).encode(
        theta=alt.Theta("val:Q",stack=True),
        color=alt.Color("k:N",scale=alt.Scale(domain=["v","rest"],range=[color,"#E6ECE8"]),legend=None),
        order=alt.Order("k:N",sort="descending"))
    txt=alt.Chart(pd.DataFrame({"t":[f"{v:.0f}%"]})).mark_text(
        fontSize=27,fontWeight="bold",color=GREEN_DEEP).encode(text="t:N")
    return (arc+txt).properties(height=155)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    df=pd.read_csv(DATA_FILE,keep_default_na=False)
    df["JoinYear"]=df["Date Joined PDO"].str[:4].astype(int)
    for h in ["Readiness 1-2 yrs","Readiness 3 yrs","Readiness 5 yrs"]:
        df[h+" %"]=df[h].apply(lambda s:int(re.search(r"(\d+)%",str(s)).group(1)) if re.search(r"(\d+)%",str(s)) else np.nan)
    df["Sadara_n"]=pd.to_numeric(df["Sadara Survey (people leadership)"],errors="coerce")
    df["S360"]=pd.to_numeric(df["360 leadership development survey"],errors="coerce")
    df["EQ"]=(df["Big5 Emotional Stability"]*.3+df["Big5 Agreeableness"]*.25+
              df["HPI Interpersonal Sensitivity"]*.25+df["HPI Adjustment"]*.2).round(0)
    df["perf"]=df["IPF 2026"].map({"EE":3,"AE":2,"MM":1,"":0}).fillna(0)
    df["Name"]=df["First Name"]+" "+df["Last Name"]
    return df

df=load_data()
GROUP_ORDER=["Director","1","2","3","4","5","6"]
GLBL={"Director":"Director","1":"G1 · Manager","2":"G2 · Head","3":"G3 · Lead",
      "4":"G4 · Senior","5":"G5 · Junior","6":"G6 · Graduate"}

def comp_health(frame):
    cnt={"Knowledge":0,"Skill":0,"Mastery":0}
    for cba in frame["Competence Based Assessment"]:
        for part in str(cba).split(";"):
            if ":" in part:
                lvl=part.rsplit(":",1)[1].strip()
                if lvl in cnt: cnt[lvl]+=1
    tot=sum(cnt.values()) or 1
    return round((cnt["Mastery"]*100+cnt["Skill"]*60+cnt["Knowledge"]*30)/tot), round(cnt["Knowledge"]/tot*100), cnt

def person_strengths_gaps(r):
    strengths=[]; gaps=[]
    for part in str(r["Competence Based Assessment"]).split(";"):
        if ":" in part:
            n,l=part.rsplit(":",1); n,l=n.strip(),l.strip()
            if l=="Mastery": strengths.append(n)
            elif l=="Knowledge": gaps.append(n)
    if r["Sadara_n"]==r["Sadara_n"] and r["Sadara_n"]>=80: strengths.append("Strong people-leadership (Sadara)")
    if r["HPI Ambition"]>=65: strengths.append("High drive & ambition")
    if r["EQ"]>=65: strengths.append("High emotional intelligence")
    if r["Sadara_n"]==r["Sadara_n"] and r["Sadara_n"]<60: gaps.append("People-leadership (Sadara) below par")
    if r["HDS Skeptical"]>=60: gaps.append("Can be overly skeptical (derailer)")
    if r["Readiness 1-2 yrs %"]<60: gaps.append("Readiness for next role still maturing")
    return strengths[:3] or ["Solid all-round contributor"], gaps[:3] or ["No major gaps flagged"]

# =========================================================
# CLAUDE
# =========================================================
def get_client():
    try: from anthropic import Anthropic
    except Exception: return None
    key=None
    try: key=st.secrets.get("ANTHROPIC_API_KEY")
    except Exception: pass
    key=key or os.environ.get("ANTHROPIC_API_KEY")
    return Anthropic(api_key=key) if key else None
client=get_client()
def ask(system,user,mx=1200):
    if client is None: return None
    m=client.messages.create(model=MODEL,max_tokens=mx,system=system,
                             messages=[{"role":"user","content":user}])
    return m.content[0].text

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown(H(f"""<div class="sb-brand">{LOGO}
<h3>Workforce Intelligence</h3><span>PDO Talent Hub</span></div>"""),unsafe_allow_html=True)
page=st.sidebar.radio("nav",["📊  Executive Summary","🎯  Talent Profile",
                             "🤝  Dream Team","🗂️  Source File – DB"],label_visibility="collapsed")
st.sidebar.markdown(
    "<div style='font-size:0.72rem;color:#BFE6CC;margin:-4px 4px 0 6px;font-style:italic'>"
    "Simulation data · source file for reference</div>", unsafe_allow_html=True)
st.sidebar.markdown("<br>",unsafe_allow_html=True)
if client:
    st.sidebar.success("🟢 AI Advisor connected")
else:
    st.sidebar.warning("⚠️ AI offline — add ANTHROPIC_API_KEY")

# =========================================================
# PAGE 1 — EXECUTIVE SUMMARY
# =========================================================
if "Executive" in page:
    header()
    f1,f2,f3=st.columns([1.2,1.4,1])
    fdir=f1.selectbox("🏢 Directorate",["All Directorates"]+sorted(df["Directorate"].unique()))
    yr=f2.slider("📅 Joined PDO — year range",1995,2026,(1995,2026))
    grp=f3.multiselect("🎚️ Job Group",GROUP_ORDER,default=GROUP_ORDER,format_func=lambda g:GLBL[g])
    d=df.copy()
    if fdir!="All Directorates": d=d[d["Directorate"]==fdir]
    d=d[(d["JoinYear"]>=yr[0])&(d["JoinYear"]<=yr[1])]
    if grp: d=d[d["Job Group"].astype(str).isin(grp)]

    health,gap,_=comp_health(d)
    k=st.columns(6)
    kpi(k[0],"Headcount",f"{len(d):,}","active profiles")
    kpi(k[1],"Omanisation",f"{d['Nationality'].eq('Omani').mean()*100:.0f}%","of workforce","alt2")
    kpi(k[2],"Female",f"{d['Gender'].eq('F').mean()*100:.0f}%","diversity","alt2")
    kpi(k[3],"High Potential",f"{d['Potential Index Band'].isin(['High Potential','Expert Track']).mean()*100:.0f}%","talent pool","alt")
    kpi(k[4],"Avg PDO Exp",f"{d['PDO Experience (yrs)'].mean():.0f} yr","tenure","alt3")
    kpi(k[5],"Competency Health",f"{health}/100","capability")

    st.markdown("<br>",unsafe_allow_html=True)
    sec("🧭 Organisation Status — at a glance")
    lead_pool=d[d["Job Group"].astype(str).isin(["1","2","3"])]
    lead_ready=lead_pool["Readiness 1-2 yrs %"].mean() if len(lead_pool) else 0
    sad=d["Sadara_n"].dropna()
    g=st.columns(4)
    titles=["Competency Health","Capability Gap","Leadership Readiness","People-Leadership (Sadara)"]
    vals=[health,gap,lead_ready,sad.mean() if len(sad) else 0]
    goods=[True,False,True,True]
    for i in range(4):
        with g[i]:
            st.altair_chart(gauge(vals[i],goods[i]),use_container_width=True)
            html(f'<div class="gcap">{titles[i]}</div>')

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns([1.4,1])
    with c1:
        sec("👥 Workforce Structure")
        vc=d["Job Group"].astype(str).value_counts().reindex(GROUP_ORDER).fillna(0)
        pdata=pd.DataFrame({"Group":[GLBL[g] for g in GROUP_ORDER],"Count":vc.values.astype(int)})
        ch=alt.Chart(pdata).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Count:Q",title=None),
            y=alt.Y("Group:N",sort=[GLBL[g] for g in GROUP_ORDER],title=None),
            color=alt.Color("Count:Q",scale=alt.Scale(range=[LIME,GREEN_DARK]),legend=None),
            tooltip=["Group","Count"])
        lbl=ch.mark_text(align="left",dx=4,color=GREEN_DEEP,fontWeight="bold").encode(text="Count:Q")
        st.altair_chart((ch+lbl).properties(height=300),use_container_width=True)
    with c2:
        sec("🌍 Nationality & Gender")
        def donut(series):
            dd=series.value_counts().reset_index(); dd.columns=["cat","n"]
            return alt.Chart(dd).mark_arc(innerRadius=45,cornerRadius=3).encode(
                theta="n:Q",color=alt.Color("cat:N",scale=alt.Scale(range=GREEN_SCHEME),
                legend=alt.Legend(orient="bottom",title=None)),tooltip=["cat","n"]).properties(height=140)
        st.altair_chart(donut(d["Nationality"]),use_container_width=True)
        st.altair_chart(donut(d["Gender"]),use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    cm,ct={}, {}
    for cba in d["Competence Based Assessment"]:
        for part in str(cba).split(";"):
            if ":" in part:
                n,l=part.rsplit(":",1); n,l=n.strip(),l.strip()
                ct[n]=ct.get(n,0)+1; cm[n]=cm.get(n,0)+{"Mastery":3,"Skill":2,"Knowledge":1}.get(l,0)
    idx={c:cm[c]/ct[c] for c in cm if ct[c]>30}
    ranked=sorted(idx.items(),key=lambda x:x[1],reverse=True)
    THEME_S=["Transformational Leadership","Operational Excellence","Technical Mastery","Safety Culture","Delivery & Execution"]
    THEME_G=["Digital & AI Transformation","Agile Ways of Working","Data & Analytics","Commercial Acumen","Innovation & Design"]
    s1,s2,s3=st.columns(3)
    with s1:
        sec("💪 Top Strengths")
        for (c,v),t in zip(ranked[:5],THEME_S): barrow(st,t,v/3*100,GREEN,f"{v/3*100:.0f}%")
    with s2:
        sec("🎯 Improvement Areas")
        for (c,v),t in zip(ranked[-5:][::-1],THEME_G): barrow(st,t,v/3*100,GOLD,f"{v/3*100:.0f}%")
    with s3:
        sec("🌟 Leadership Bench")
        ls=[("Strategic Thinking",lead_ready),("People Leadership",sad.mean() if len(sad) else 0),
            ("Change Leadership",max(30,lead_ready-8)),("Decision Making",min(95,lead_ready+6)),
            ("Stakeholder Mgmt",max(35,lead_ready-3))]
        for n,val in ls:
            col=GREEN if val>=70 else (GOLD if val>=50 else RED)
            barrow(st,n,val,col,f"{val:.0f}%")

    st.markdown("<br>",unsafe_allow_html=True)
    sec("🪜 Succession Readiness — effort to build the next layer")
    nxt={"Director":"→ Exec Director","1":"→ Director","2":"→ Snr Manager","3":"→ Head",
         "4":"→ Team Lead","5":"→ Senior","6":"→ Officer"}
    rows=[]
    for gg in GROUP_ORDER:
        sub=d[d["Job Group"].astype(str)==gg]
        if len(sub):
            rows.append({"Transition":f"{GLBL[gg]} {nxt[gg]}","Avg":round(sub['Readiness 1-2 yrs %'].mean()),
                         "ReadyNow":round((sub['Readiness 1-2 yrs %']>=80).mean()*100),"Pool":len(sub)})
    cA,cB=st.columns([1.25,1])
    with cA:
        rdf=pd.DataFrame(rows)
        base=alt.Chart(rdf).encode(y=alt.Y("Transition:N",sort=list(rdf["Transition"]),title=None,
             axis=alt.Axis(labelLimit=240,labelFontSize=12,labelColor=GREEN_DEEP)))
        bar=base.mark_bar(cornerRadius=7,size=20).encode(
            x=alt.X("Avg:Q",title="Avg readiness %",scale=alt.Scale(domain=[0,100])),
            color=alt.Color("Avg:Q",scale=alt.Scale(domain=[30,60,90],range=[RED,GOLD,GREEN]),legend=None),
            tooltip=["Transition","Avg","ReadyNow","Pool"])
        st.altair_chart((bar).properties(height=300),use_container_width=True)
    with cB:
        for r in rows:
            col=GREEN if r["Avg"]>=70 else (GOLD if r["Avg"]>=50 else RED)
            html(f"""<div class="rowitem" style="border-left-color:{col}"><b>{r['Transition']}</b>
            <span style="float:right;background:{col};color:#fff;padding:1px 9px;border-radius:12px;font-size:.76rem">{r['ReadyNow']}% ready</span>
            <div style="color:{SLATE};font-size:.76rem;margin-top:3px">pool {r['Pool']:,} · avg {r['Avg']}%</div></div>""")

    if client:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🧠 Generate executive narrative"):
            payload={"headcount":len(d),"omanisation":f"{d['Nationality'].eq('Omani').mean()*100:.0f}%",
                     "health":health,"gap":gap,"leadership_readiness":round(lead_ready),
                     "strengths":THEME_S,"improvements":THEME_G,"succession":rows}
            out=ask("PDO HR strategy advisor. Crisp executive narrative <180 words: strengths, gaps, "
                    "succession risk, 3 priority actions. Synthetic data.",json.dumps(payload))
            html(f'<div class="card">{out}</div>')

# =========================================================
# PAGE 2 — TALENT PROFILE
# =========================================================
elif "Talent" in page:
    header()
    html(f'<div style="color:{SLATE};font-size:.95rem;margin:2px 0 8px 2px">🎯 <b>Find the right person, instantly.</b> '
         f'Type the role or talent you need — the Hub scans your whole workforce and ranks the best fits with full reasoning.</div>')
    q=st.text_area("Search",placeholder="e.g. Best successor for HR (People & Culture) Director with strong leadership and change capability…",
                   label_visibility="collapsed",height=90)
    c2,c3=st.columns([1,1])
    fdir=c2.selectbox("🏢 Directorate filter",["All (whole org)"]+sorted(df["Directorate"].unique()))
    topn=c3.slider("👥 Candidates to compare",3,10,6)
    go=st.button("🚀 Find best-fit talent",type="primary")

    # Target role -> eligible feeder groups (who can realistically fill it)
    # Longest phrases first so "senior manager"/"team lead" match before "manager"/"lead".
    TARGET_RULES=[("executive director",["Director"]),
        ("senior manager",["1","2"]),("team lead",["3","4"]),
        ("director",["1"]),            # Director drawn ONLY from G1 Managers
        ("manager",["1","2"]),
        ("head",["2","3"]),
        ("lead",["3","4"]),            # Lead (G3) drawn from G3 + G4
        ("senior",["4","5"]),
        ("officer",["5","6"])]

    # words to ignore when extracting the *subject* of the search
    STOP=set(("best successor succession for with without strong the a an to and or of who whom "
              "person people talent talents candidate candidates find need needs looking role roles "
              "position positions director manager head lead leader senior officer graduate executive "
              "leadership capable fit good great top someone our new next in on at as be is are i want "
              "please show me who someone experienced experience skills skill able ready readiness").split())
    def keywords(text):
        return [t for t in re.findall(r"[a-z]+",text.lower()) if t not in STOP and len(t)>2]

    if go and q:
        ql=q.lower(); rule=None; tname="role"
        for key,groups in TARGET_RULES:
            if key in ql: rule=groups; tname=key.title(); break

        d=df.copy()
        if fdir!="All (whole org)": d=d[d["Directorate"]==fdir]
        # HIERARCHY: hard-filter to eligible feeder groups
        if rule: d=d[d["Job Group"].astype(str).isin(rule)]
        d=d.copy()

        # ---- Understand PERFORMANCE preference in the query ----
        # e.g. "not EE", "wasn't EE", "no EE", "exclude EE" -> drop the always-EE crowd.
        ipfmap={"EE":3,"AE":2,"MM":1,"":np.nan}
        exclude_ee = bool(re.search(r"(not|no|without|non|exclude|didn'?t|wasn'?t|never)\s+.*ee", ql)) or "not ee" in ql
        # 3-year performance trajectory (2024-26), ignoring blanks
        def perf3(r):
            vals=[ipfmap.get(r[f]) for f in ["IPF 2024","IPF 2025","IPF 2026"]]
            vals=[v for v in vals if v==v]  # drop NaN
            return np.mean(vals) if vals else 1.0
        d["perf3"]=d.apply(perf3,axis=1)
        d["ee_years"]=d.apply(lambda r:sum(1 for f in ["IPF 2025","IPF 2026"] if r[f]=="EE"),axis=1)
        if exclude_ee:
            # user explicitly wants NON-EE: drop anyone EE in either of the last 2 years
            d=d[d["ee_years"]==0].copy()

        # did the user explicitly name a directorate? (e.g. "...in HR")
        named_dir=None
        DIR_ALIASES={"Information & Digital (IDD)":["idd","it","digital","information"],
                     "People & Culture (HR)":["hr","people","culture","human"],
                     "Supply Chain (CP)":["supply","procurement","logistics","cp"],
                     "Finance":["finance","financial"],"HSE":["hse","safety","health"],
                     "Engineering & Projects":["engineering","projects"],"Operations":["operations","production"],
                     "Exploration":["exploration","subsurface","geoscience"],
                     "Legal & Corporate":["legal","corporate","compliance"]}
        for dn,al in DIR_ALIASES.items():
            if any(a in ql for a in al): named_dir=dn; break

        # ---- CONTENT RELEVANCE (whole org). Use PRESENCE per field so one
        # repeated word (e.g. a function name) can't dominate. ----
        kws=keywords(q)
        fields=["Directorate","Parent Function","Current Job Title","Competence Based Assessment",
                "Trainings Completed","Degree","Certificates"]
        if kws:
            def relev(r):
                score=0
                for f in fields:
                    txt=str(r[f]).lower()
                    for k in kws:
                        if k in txt: score+=1   # presence, capped per field -> no inflation
                return score
            d["relevance"]=d.apply(relev,axis=1)
        else:
            d["relevance"]=0

        d["Ready%"]=d["Readiness 1-2 yrs %"].fillna(0)
        # Balanced quality: performance no longer dominates. Uses 3-yr trajectory (not just 2026),
        # so AE and mixed MM->AE->EE journeys compete fairly with straight-EE people.
        quality=(d["perf3"]*6           # was perf*10 (EE-heavy); now smoother & lighter
                 + d["Ready%"]*.55
                 + d["Sadara_n"].fillna(60)*.25
                 + d["S360"].fillna(65)*.18
                 + d["EQ"].fillna(55)*.12)
        d["fitraw"]=d["relevance"]*18 + quality
        if named_dir is not None:
            d.loc[d["Directorate"]==named_dir,"fitraw"]+=25   # honour a named directorate, don't restrict to it

        matched=int((d["relevance"]>0).sum()) if kws else len(d)

        # ---- DIVERSITY: build a MIXED shortlist across the org (cap per directorate) ----
        ranked=d.sort_values("fitraw",ascending=False)
        cap=max(2, int(np.ceil(topn/3)))       # no directorate takes more than ~1/3
        picked=[]; per={}
        for _,r in ranked.iterrows():
            dn=r["Directorate"]
            if per.get(dn,0)<cap:
                picked.append(r); per[dn]=per.get(dn,0)+1
            if len(picked)>=topn: break
        if len(picked)<topn:  # top up if caps left us short
            for _,r in ranked.iterrows():
                if not any(r["Company Number"]==p["Company Number"] for p in picked):
                    picked.append(r)
                if len(picked)>=topn: break
        d=pd.DataFrame(picked).reset_index(drop=True)

        lo,hi=d["fitraw"].min(),d["fitraw"].max()
        d["Match"]=((d["fitraw"]-lo)/(hi-lo+1e-9)*26+72).round(0)
        d.loc[0,"Match"]=max(d.loc[0,"Match"],95)

        if rule:
            st.info(f"🔎 **{tname}** roles are filled from **{', '.join(GLBL[g] for g in rule)}** — "
                    f"searched **across the whole organisation** for *{', '.join(kws) or 'best fit'}*"
                    + (f", with a boost for **{named_dir}**" if named_dir else "")+".")
        st.success(f"Scanned the whole org · **{matched}** relevant profiles · showing a **mixed** top "
                   f"**{len(d)}** from **{d['Directorate'].nunique()} directorates**.")
        if exclude_ee:
            st.warning("⚙️ Filter applied: **excluding anyone rated EE** in the last two years — "
                       "surfacing strong **AE / mixed-trajectory** performers instead.")

        # ---- 1) Candidate list with match % ----
        sec("📋 Potential Candidates (ranked by match)")
        for i,r in d.iterrows():
            crown="👑 " if i==0 else ""
            col=GREEN if r["Match"]>=90 else (GOLD if r["Match"]>=82 else SLATE)
            html(f"""<div class="rowitem" style="border-left-color:{col}">
            <b>{crown}{r['Name']}</b> <span class="pill s">{GLBL[str(r['Job Group'])]}</span>
            <span style="color:{SLATE};font-size:.8rem">{r['Current Job Title']} · {r['Directorate']}</span>
            <span style="float:right;font-weight:800;color:{col};font-size:1.05rem">{r['Match']:.0f}% match</span>
            <div class="bar"><div style="background:{col};width:{r['Match']}%"></div></div></div>""")

        st.markdown("<br>",unsafe_allow_html=True)
        # ---- 2) Per-candidate: readiness, last 3y IPF, strengths, gaps ----
        sec("🔎 Candidate Detail — readiness, performance trend, strengths & gaps")
        for i,r in d.iterrows():
            s,g=person_strengths_gaps(r)
            ipfs="".join(f'<span class="pill {"g" if v=="EE" else ("y" if v=="AE" else "s")}">{y}: {v or "—"}</span>'
                         for y,v in [("2024",r["IPF 2024"]),("2025",r["IPF 2025"]),("2026",r["IPF 2026"])])
            sada=f"{r['Sadara_n']:.0f}" if pd.notna(r['Sadara_n']) else "N/A"
            strg="".join(f'<span class="pill g">✔ {x}</span>' for x in s)
            gapg="".join(f'<span class="pill r">▲ {x}</span>' for x in g)
            html(f"""<div class="card">
            <b style="color:{GREEN_DEEP};font-size:1.02rem">{'👑 ' if i==0 else ''}{r['Name']}</b>
            <span style="float:right;color:{SLATE};font-size:.82rem">Readiness (1–2y): <b style="color:{GREEN_DARK}">{r['Ready%']:.0f}%</b> · Sadara {sada} · EQ {r['EQ']:.0f}</span>
            <div style="margin:8px 0 4px 0"><span style="color:{SLATE};font-size:.8rem">Last 3-yr IPF:</span> {ipfs}</div>
            <div style="margin-top:6px">{strg}</div><div style="margin-top:4px">{gapg}</div></div>""")

        st.markdown("<br>",unsafe_allow_html=True)
        # ---- 3) Best fit selection deep-dive ----
        top=d.loc[0]
        sec(f"🏆 Best-Fit Selection — {top['Name']}")
        s,g=person_strengths_gaps(top)
        kc=st.columns(5)
        kpi(kc[0],"Match",f"{top['Match']:.0f}%",f"for {tname}")
        kpi(kc[1],"Readiness",f"{top['Ready%']:.0f}%","1–2 yr horizon","alt2")
        kpi(kc[2],"IPF 2026",top["IPF 2026"],"performance","alt")
        kpi(kc[3],"Sadara",f"{top['Sadara_n']:.0f}" if pd.notna(top['Sadara_n']) else "N/A","people-leadership","alt3")
        kpi(kc[4],"360° / EQ",f"{top['S360']:.0f} / {top['EQ']:.0f}","leadership signals")

        st.markdown("<br>",unsafe_allow_html=True)
        cc1,cc2=st.columns([1,1])
        with cc1:
            sec("🧠 Hogan / Behavioural Signals")
            psy=[("Ambition (HPI)",top["HPI Ambition"]),("Adjustment (HPI)",top["HPI Adjustment"]),
                 ("Prudence (HPI)",top["HPI Prudence"]),("Dominance (DISC)",top["DISC Dominance"]),
                 ("Conscientiousness",top["Big5 Conscientiousness"]),("Learning Approach",top["HPI Learning Approach"])]
            for n,val in psy:
                col=GREEN if val>=60 else (GOLD if val>=45 else RED)
                barrow(st,n,val,col,f"{val:.0f}")
        with cc2:
            sec("📈 3-Year Performance & Readiness")
            trend=pd.DataFrame({"Year":["2024","2025","2026"],
                "Score":[{"EE":3,"AE":2,"MM":1,"":0}.get(top[f"IPF {y}"],0) for y in ["2024","2025","2026"]]})
            ch=alt.Chart(trend).mark_line(point=alt.OverlayMarkDef(size=90,color=GREEN),color=GREEN,strokeWidth=3).encode(
                x=alt.X("Year:N",title=None),
                y=alt.Y("Score:Q",scale=alt.Scale(domain=[0,3]),
                        axis=alt.Axis(values=[1,2,3],labelExpr="datum.value==3?'EE':datum.value==2?'AE':'MM'",title=None)),
                tooltip=["Year","Score"]).properties(height=150)
            st.altair_chart(ch,use_container_width=True)
            rl=pd.DataFrame({"Horizon":["1–2 yr","3 yr","5 yr"],
                "Coverage":[top["Readiness 1-2 yrs %"],top["Readiness 3 yrs %"],top["Readiness 5 yrs %"]]})
            ch=alt.Chart(rl).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Coverage:Q",scale=alt.Scale(domain=[0,100]),title="Competency coverage %"),
                y=alt.Y("Horizon:N",title=None),
                color=alt.Color("Coverage:Q",scale=alt.Scale(domain=[30,60,90],range=[RED,GOLD,GREEN]),legend=None),
                tooltip=["Horizon","Coverage"]).properties(height=120)
            st.altair_chart(ch,use_container_width=True)

        # AI verdict — forced consistent with #1
        if client:
            cols=["Name","Directorate","Current Job Title","Job Group","IPF 2024","IPF 2025","IPF 2026",
                  "Ready%","Sadara_n","S360","EQ","Match"]
            out=ask(f"You are the PDO Talent Advisor. The SELECTED best-fit is **{top['Name']}** (candidate #1) — "
                    f"you MUST endorse this person as the top pick and stay consistent with the ranking order given. "
                    f"In <160 words explain WHY {top['Name']} is the right choice (cite match %, readiness, Sadara, "
                    f"360°, IPF trend, EQ) and name 2 runners-up in one line each. Synthetic data.",
                    f"Role: {q}\nRanked candidates (row 0 = selected):\n{d[cols].to_json(orient='records')}")
            sec("💡 Advisor Verdict")
            html(f'<div class="card">{out}</div>')

        sec("📋 Shortlisted Data Used")
        show=["Company Number","Name","Directorate","Current Job Title","Job Group","Match",
              "IPF 2026","Ready%","Sadara_n","S360","EQ","Potential Index Band"]
        st.dataframe(d[show].rename(columns={"Sadara_n":"Sadara","S360":"360°"}),hide_index=True,use_container_width=True)

# =========================================================
# PAGE 3 — DREAM TEAM
# =========================================================
elif "Dream" in page:
    header()
    html(f'<div style="color:{SLATE};font-size:.95rem;margin:2px 0 8px 2px">🤝 <b>Build your ideal task force.</b> '
         f'Describe the mission — the Hub scans all 11,000 people and assembles a balanced, cross-functional team with a recommended lead.</div>')
    scope=st.text_area("Mission",placeholder="e.g. Stand up a Microsoft Fabric data platform task force to unify PDO reporting…",
                       label_visibility="collapsed",height=90)
    c1,c2=st.columns([1,1])
    size=c1.slider("👥 Team size",4,12,7)
    lead_dir=c2.selectbox("🏢 Primary directorate (weighted)",["Auto-detect"]+sorted(df["Directorate"].unique()))
    go=st.button("🛠️ Build Dream Team",type="primary")

    if go and scope:
        s=scope.lower()
        # word-boundary keyword matcher (fixes "ai" matching inside "chain")
        def has(word): return re.search(r"\b"+re.escape(word)+r"\b", s) is not None
        primary=lead_dir
        if lead_dir=="Auto-detect":
            kwm={"Supply Chain (CP)":["supply","chain","procurement","procure","sourcing","supplier","logistics","inventory","warehouse","contract","contracts","category","tender","vendor"],
                 "Information & Digital (IDD)":["fabric","data","cyber","digital","software","cloud","system","analytics","ai","it","power bi","report","application"],
                 "Finance":["finance","budget","cost","treasury","audit","accounting"],
                 "HSE":["safety","hse","environment","incident","emergency"],
                 "Engineering & Projects":["project","engineering","design","construction","well","projects"],
                 "Operations":["production","operations","reservoir","maintenance","asset"],
                 "People & Culture (HR)":["hr","talent","training","recruit","learning","workforce"],
                 "Exploration":["exploration","subsurface","seismic","geoscience","petrophysics"],
                 "Legal & Corporate":["legal","compliance","governance","regulatory"]}
            best=None;bn=0
            for dn,kws in kwm.items():
                n=sum(1 for w in kws if has(w))
                if n>bn:best,bn=dn,n
            primary=best or "Information & Digital (IDD)"

        d=df.copy()
        # ---- CONTENT RELEVANCE: does this person actually work in this domain? ----
        kws=[w for w in re.findall(r"[a-z]+",s) if len(w)>3 and w not in
             {"team","build","want","process","enhancement","improve","improvement","create","need","help","with","that","this","have","from","across","best","people","member","members","task","force","group","project"}]
        d["dtext"]=(d["Directorate"]+" "+d["Parent Function"]+" "+d["Current Job Title"]+" "+
                    d["Competence Based Assessment"]+" "+d["Trainings Completed"]).str.lower()
        d["relevance"]=d["dtext"].apply(lambda t:sum(1 for k in kws if k in t)) if kws else 0

        d["teamwork"]=d["Big5 Agreeableness"]*.5+d["DISC Influence"]*.3+d["Big5 Conscientiousness"]*.2
        d["fit"]=(d["relevance"]*14 + d["perf"]*8 + d["PDO Experience (yrs)"]*.8 +
                  d["Sadara_n"].fillna(60)*.4 + d["teamwork"]*.3)
        d.loc[d["Directorate"]==primary,"fit"]+=20   # domain experts lead the core

        # Core = ~60% from the primary (domain) directorate; rest cross-functional
        npri=max(2,round(size*.6))
        core=d[d["Directorate"]==primary].nlargest(min(npri,size),"fit")
        others=d[d["Directorate"]!=primary].nlargest(size-len(core),"fit")
        team=pd.concat([core,others]).sort_values("fit",ascending=False).reset_index(drop=True)
        lead=team.loc[0]

        st.success(f"🎯 Domain: **{primary}** · {len(core)} core domain experts + "
                   f"{len(team)-len(core)} cross-functional · {size}-member team assembled.")
        k=st.columns(4)
        kpi(k[0],"Team Lead",lead["Name"],lead["Current Job Title"])
        kpi(k[1],"Avg Sadara",f"{team['Sadara_n'].dropna().mean():.0f}" if team['Sadara_n'].notna().any() else "—","behaviour","alt2")
        kpi(k[2],"Avg PDO Exp",f"{team['PDO Experience (yrs)'].mean():.0f} yr","depth","alt")
        kpi(k[3],"Directorates",f"{team['Directorate'].nunique()}","cross-functional","alt3")

        # ---- Team structure: lead on top, members below ----
        st.markdown("<br>",unsafe_allow_html=True)
        sec("🏗️ Team Structure")
        lsada=f"{lead['Sadara_n']:.0f}" if pd.notna(lead['Sadara_n']) else "N/A"
        html(f"""<div class="team-lead"><div class="n">👑 {lead['Name']}</div>
        <div class="r">{lead['Current Job Title']} · {lead['Directorate']}</div>
        <div class="r">IPF {lead['IPF 2026']} · Sadara {lsada} · {lead['PDO Experience (yrs)']}y</div></div>
        <div class="connector"></div>""")
        members=team.iloc[1:]
        per=st.columns(min(4,max(1,len(members))))
        for i,(_,m) in enumerate(members.iterrows()):
            sada=f"{m['Sadara_n']:.0f}" if pd.notna(m['Sadara_n']) else "N/A"
            per[i%len(per)].markdown(H(f"""<div class="member"><div class="n">{m['Name']}</div>
            <div class="t">{m['Current Job Title']}<br>{m['Directorate']}</div>
            <div class="m">🎖️ IPF {m['IPF 2026']} · 🤝 {sada} · 🧭 {m['PDO Experience (yrs)']}y</div></div>"""),
            unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        cc1,cc2=st.columns([1,1.1])
        with cc1:
            sec("🧩 Directorate Mix")
            mix=team["Directorate"].value_counts().reset_index();mix.columns=["Directorate","n"]
            ch=alt.Chart(mix).mark_arc(innerRadius=55,cornerRadius=3).encode(
                theta="n:Q",color=alt.Color("Directorate:N",scale=alt.Scale(range=GREEN_SCHEME),
                legend=alt.Legend(orient="bottom",title=None)),tooltip=["Directorate","n"]).properties(height=280)
            st.altair_chart(ch,use_container_width=True)
        with cc2:
            sec("⚖️ Member Fit & Behaviour")
            comp=team[["Name","Sadara_n","teamwork","PDO Experience (yrs)"]].copy()
            comp["Sadara_n"]=comp["Sadara_n"].fillna(60)
            melt=comp.melt("Name",var_name="Metric",value_name="Score")
            melt["Metric"]=melt["Metric"].map({"Sadara_n":"Sadara","teamwork":"Teamwork","PDO Experience (yrs)":"Experience"})
            ch=alt.Chart(melt).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Score:Q"),y=alt.Y("Name:N",sort="-x",title=None),
                color=alt.Color("Metric:N",scale=alt.Scale(range=[GREEN,LIME,GOLD]),legend=alt.Legend(orient="top",title=None)),
                tooltip=["Name","Metric","Score"]).properties(height=280)
            st.altair_chart(ch,use_container_width=True)

        sec("👤 Member Insights")
        cols=st.columns(2)
        for i,(_,m) in enumerate(team.iterrows()):
            role="👑 LEAD" if i==0 else "Member"
            sada=f"{m['Sadara_n']:.0f}" if pd.notna(m['Sadara_n']) else "N/A"
            cols[i%2].markdown(H(f"""<div class="rowitem"><b>{m['Name']}</b> <span class="pill g">{role}</span>
            <span style="float:right;color:{SLATE};font-size:.78rem">{m['Directorate']}</span><br>
            <span style="color:{SLATE};font-size:.82rem">{m['Current Job Title']} · {GLBL[str(m['Job Group'])]}</span><br>
            <span style="font-size:.8rem">🎖️ IPF {m['IPF 2026']} · 🤝 Sadara {sada} · 🧭 {m['PDO Experience (yrs)']}y · 💡 EQ {m['EQ']:.0f}</span></div>"""),
            unsafe_allow_html=True)

        if client:
            show=["Name","Directorate","Current Job Title","Job Group","IPF 2026","Sadara_n","PDO Experience (yrs)"]
            out=ask("PDO Dream Team builder. <130 words: why THIS team succeeds (complementary strengths, "
                    "cross-functional coverage). Positive only. Do NOT mention weaknesses or failure. Synthetic.",
                    f"Mission: {scope}\nLead: {lead['Name']}\nTeam:\n{team[show].to_json(orient='records')}")
            sec("💡 Why this team works")
            html(f'<div class="card">{out}</div>')
            out2=ask("PDO Dream Team builder. <70 words: why the named LEAD is the right leader for this mission "
                     "(cite Sadara, IPF, experience). Positive, concise. Synthetic.",
                     f"Mission: {scope}\nLead: {lead['Name']} · {lead['Current Job Title']} · "
                     f"Sadara {lead['Sadara_n']} · IPF {lead['IPF 2026']} · {lead['PDO Experience (yrs)']}y")
            sec("👑 Why this lead")
            html(f'<div class="card">{out2}</div>')

# =========================================================
# PAGE 4 — SOURCE FILE / DATABASE (full 11k visibility)
# =========================================================
elif "Source File" in page:
    header()
    html(f'<div style="color:{SLATE};font-size:.95rem;margin:2px 0 8px 2px">🗂️ <b>Source data — full workforce database.</b> '
         f'The complete set of <b>{len(df):,}</b> synthetic employee records powering this Hub. Search, filter, sort and export for full transparency.</div>')

    # KPI strip
    k=st.columns(4)
    kpi(k[0],"Total Records",f"{len(df):,}","employees")
    kpi(k[1],"Data Fields",f"{df.shape[1]}","columns","alt")
    kpi(k[2],"Directorates",f"{df['Directorate'].nunique()}","business areas","alt2")
    kpi(k[3],"Data Source","CSV","phase 1 (DB-ready)","alt3")

    st.markdown("<br>",unsafe_allow_html=True)
    # Filters
    sec("🔎 Search & Filter")
    c1,c2,c3=st.columns([1.4,1,1])
    txt=c1.text_input("Search by name, job title, competency, degree…","")
    fdir=c2.multiselect("Directorate",sorted(df["Directorate"].unique()))
    fgrp=c3.multiselect("Job Group",GROUP_ORDER,format_func=lambda g:GLBL[g])

    view=df.copy()
    if fdir: view=view[view["Directorate"].isin(fdir)]
    if fgrp: view=view[view["Job Group"].astype(str).isin(fgrp)]
    if txt:
        t=txt.lower()
        cols_search=["First Name","Last Name","Current Job Title","Directorate","Parent Function",
                     "Competence Based Assessment","Degree","Certificates","Trainings Completed"]
        mask=view[cols_search].apply(lambda r:t in " ".join(map(str,r.values)).lower(),axis=1)
        view=view[mask]

    st.caption(f"Showing **{len(view):,}** of {len(df):,} records.")
    # Drop helper columns created at load-time so the raw source is shown
    hide=["JoinYear","Readiness 1-2 yrs %","Readiness 3 yrs %","Readiness 5 yrs %",
          "Sadara_n","S360","EQ","perf","Name","stext","dtext","relevance"]
    show=[c for c in df.columns if c not in hide]
    st.dataframe(view[show],hide_index=True,use_container_width=True,height=560)

    st.download_button("⬇️ Download filtered data (CSV)",
                       view[show].to_csv(index=False).encode("utf-8"),
                       "pdo_workforce_data.csv","text/csv")

    # Individual record viewer
    st.markdown("<br>",unsafe_allow_html=True)
    sec("👤 Inspect a single record")
    if len(view):
        opts=(view["First Name"]+" "+view["Last Name"]+"  ·  "+view["Current Job Title"]+
              "  ·  #"+view["Company Number"].astype(str)).tolist()
        pick=st.selectbox("Select an employee",opts)
        num=int(pick.split("#")[-1])
        person=df[df["Company Number"]==num].iloc[0]
        cc=st.columns(4)
        kpi(cc[0],"Name",f"{person['First Name']} {person['Last Name']}",person['Nationality'])
        kpi(cc[1],"Role",person['Current Job Title'][:22],GLBL[str(person['Job Group'])],"alt")
        kpi(cc[2],"Directorate",person['Directorate'][:20],person['Parent Function'],"alt2")
        kpi(cc[3],"PDO Exp",f"{person['PDO Experience (yrs)']} yr",f"joined {person['Date Joined PDO'][:4]}","alt3")
        with st.expander("📄 Full profile — all fields",expanded=True):
            rec=person[show].to_dict()
            st.dataframe(pd.DataFrame({"Field":list(rec.keys()),"Value":[str(v) for v in rec.values()]}),
                         hide_index=True,use_container_width=True,height=430)
