"""
Workforce Intelligence Hub — PDO (synthetic data)
Corporate PowerBI/PowerApps style. Green PDO theme.
Pages: Executive Summary | Talent Profile | Dream Team | Training & Upskilling
Phase 1: CSV-powered. ALL DATA SYNTHETIC.
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

# ---------- Theme palette (PDO green) ----------
GREEN      = "#00843D"
GREEN_DARK = "#005A2B"
GREEN_DEEP = "#0A3D2C"
LIME       = "#7AC143"
GOLD       = "#F2A900"
RED        = "#E24A33"
SLATE      = "#5D6D7E"
GREEN_SCHEME = [ "#0A3D2C","#005A2B","#00843D","#4CAF50","#7AC143","#B4E197"]

# =========================================================
# GLOBAL CSS — corporate look
# =========================================================
st.markdown(f"""
<style>
#MainMenu, header[data-testid="stHeader"], footer {{visibility:hidden;}}
.stApp {{ background:#EDF2EE; }}
.block-container {{ padding-top:0.5rem; padding-bottom:2rem; max-width:1500px; }}

/* ---- Top header banner ---- */
.wih-header {{
  background:linear-gradient(100deg,{GREEN_DEEP} 0%,{GREEN_DARK} 45%,{GREEN} 100%);
  border-radius:16px; padding:18px 26px; margin-bottom:18px;
  display:flex; align-items:center; gap:18px;
  box-shadow:0 6px 22px rgba(0,90,43,.28);
}}
.wih-title {{ color:#fff; font-size:2.05rem; font-weight:800; letter-spacing:.3px; margin:0; }}
.wih-sub {{ color:#CDEBD6; font-size:.95rem; margin-top:2px; }}
.wih-badge {{ margin-left:auto; background:rgba(255,255,255,.14); color:#fff;
  padding:8px 16px; border-radius:30px; font-size:.85rem; font-weight:600;
  border:1px solid rgba(255,255,255,.25);}}

/* ---- KPI cards ---- */
.kpi {{ background:#fff; border-radius:14px; padding:16px 18px; height:118px;
  box-shadow:0 3px 12px rgba(16,40,30,.08); border-top:4px solid {GREEN};
  display:flex; flex-direction:column; justify-content:center; }}
.kpi .lbl {{ color:{SLATE}; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.4px;}}
.kpi .val {{ color:{GREEN_DEEP}; font-size:2.0rem; font-weight:800; line-height:1.1; }}
.kpi .sub {{ color:#8A97A6; font-size:.78rem; }}
.kpi.alt {{ border-top-color:{GOLD}; }}
.kpi.alt2 {{ border-top-color:{LIME}; }}
.kpi.alt3 {{ border-top-color:{SLATE}; }}

/* ---- Section titles ---- */
.sec {{ font-size:1.25rem; font-weight:800; color:{GREEN_DEEP}; margin:8px 0 4px 0;
  border-left:5px solid {GREEN}; padding-left:12px; }}
.card {{ background:#fff; border-radius:14px; padding:16px 18px; margin-bottom:12px;
  box-shadow:0 3px 12px rgba(16,40,30,.07); }}
.pill {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:.8rem;
  font-weight:600; margin:3px 4px 3px 0; }}
.pill.g {{ background:#E3F3E8; color:{GREEN_DARK}; }}
.pill.y {{ background:#FDF0D5; color:#9A6B00; }}
.pill.r {{ background:#FBE3DE; color:#A93226; }}
.rowitem {{ background:#fff; border-radius:12px; padding:12px 16px; margin-bottom:8px;
  box-shadow:0 2px 8px rgba(16,40,30,.06); border-left:4px solid {GREEN}; }}

/* ---- Sidebar as nav ---- */
section[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,{GREEN_DEEP} 0%,{GREEN_DARK} 100%);
  min-width:290px !important;
}}
section[data-testid="stSidebar"] * {{ color:#EAF6EE; }}
section[data-testid="stSidebar"] .stRadio > label {{ display:none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
  border-radius:12px; padding:14px 16px; margin-bottom:10px; width:100%;
  font-size:1.05rem !important; font-weight:600; cursor:pointer;
  transition:.15s; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
  background:rgba(255,255,255,.16); transform:translateX(3px); }}
section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {{ }}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
  background:#fff; }}
/* selected state via has() (modern browsers) */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background:#ffffff; border-color:#fff; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
  color:{GREEN_DARK} !important; font-weight:800; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:1.05rem !important; }}
.sb-brand {{ text-align:center; padding:6px 0 14px 0; border-bottom:1px solid rgba(255,255,255,.15);
  margin-bottom:16px; }}
.sb-brand h3 {{ color:#fff; margin:6px 0 0 0; font-size:1.15rem; font-weight:800;}}
.sb-brand span {{ color:#BFE6CC; font-size:.78rem; }}

/* dataframe rounding */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
div[data-testid="stMetric"] {{ background:#fff; border-radius:12px; padding:12px 14px;
  box-shadow:0 3px 12px rgba(16,40,30,.07); border-top:4px solid {GREEN}; }}
.stButton>button {{ background:{GREEN}; color:#fff; border:none; border-radius:10px;
  padding:10px 20px; font-weight:700; }}
.stButton>button:hover {{ background:{GREEN_DARK}; color:#fff; }}
</style>
""", unsafe_allow_html=True)

LOGO_SVG = f"""
<svg width="54" height="54" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="2" y="2" width="60" height="60" rx="16" fill="white" fill-opacity="0.12"
 stroke="white" stroke-opacity="0.5"/>
<circle cx="32" cy="22" r="7" fill="{LIME}"/>
<circle cx="19" cy="40" r="6" fill="#fff"/>
<circle cx="45" cy="40" r="6" fill="#fff"/>
<path d="M32 29c-6 0-9 6-9 6M32 29c6 0 9 6 9 6M25 44h14" stroke="{LIME}"
 stroke-width="3" stroke-linecap="round"/>
</svg>
"""

def header():
    st.markdown(f"""
    <div class="wih-header">
      {LOGO_SVG}
      <div>
        <p class="wih-title">Workforce Intelligence Hub</p>
        <div class="wih-sub">Petroleum Development Oman · Talent & Succession Analytics</div>
      </div>
      <div class="wih-badge">🌿 11,000 profiles · synthetic</div>
    </div>
    """, unsafe_allow_html=True)

def kpi(col, label, value, sub="", cls=""):
    col.markdown(f"""<div class="kpi {cls}"><div class="lbl">{label}</div>
    <div class="val">{value}</div><div class="sub">{sub}</div></div>""",
    unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def gauge(value, title, good_high=True):
    v = float(value)
    color = GREEN if (v>=80)==good_high else (GOLD if 60<=v<80 else RED)
    if not good_high:  # for gaps, high is bad
        color = RED if v>=66 else (GOLD if v>=40 else GREEN)
    src = pd.DataFrame({"k":["v","rest"],"val":[v,100-v]})
    arc = alt.Chart(src).mark_arc(innerRadius=52, outerRadius=74, cornerRadius=4).encode(
        theta=alt.Theta("val:Q", stack=True),
        color=alt.Color("k:N", scale=alt.Scale(domain=["v","rest"],
                        range=[color,"#E6ECE8"]), legend=None),
        order=alt.Order("k:N", sort="descending"))
    txt = alt.Chart(pd.DataFrame({"t":[f"{v:.0f}%"]})).mark_text(
        fontSize=26, fontWeight="bold", color=GREEN_DEEP).encode(text="t:N")
    cap = alt.Chart(pd.DataFrame({"t":[title]})).mark_text(
        fontSize=12, color=SLATE, dy=44).encode(text="t:N")
    return (arc+txt+cap).properties(height=170)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, keep_default_na=False)
    df["JoinYear"] = df["Date Joined PDO"].str[:4].astype(int)
    for h in ["Readiness 1-2 yrs","Readiness 3 yrs","Readiness 5 yrs"]:
        df[h+" %"] = df[h].apply(lambda s:int(re.search(r"(\d+)%",str(s)).group(1))
                                 if re.search(r"(\d+)%",str(s)) else np.nan)
    df["Sadara_n"] = pd.to_numeric(df["Sadara Survey (people leadership)"], errors="coerce")
    df["S360"] = pd.to_numeric(df["360 leadership development survey"], errors="coerce")
    # Emotional/EQ composite
    df["EQ"] = (df["Big5 Emotional Stability"]*.3 + df["Big5 Agreeableness"]*.25 +
                df["HPI Interpersonal Sensitivity"]*.25 + df["HPI Adjustment"]*.2).round(0)
    ipfmap = {"EE":3,"AE":2,"MM":1,"":0}
    df["perf"] = df["IPF 2026"].map(ipfmap).fillna(0)
    return df

df = load_data()
GROUP_ORDER = ["Director","1","2","3","4","5","6"]
GLBL = {"Director":"Director","1":"G1 · Manager","2":"G2 · Head","3":"G3 · Lead",
        "4":"G4 · Senior","5":"G5 · Junior","6":"G6 · Graduate"}

def comp_health(frame):
    """Return org competency health 0-100 and gap% (share of Knowledge)."""
    cnt={"Knowledge":0,"Skill":0,"Mastery":0}
    for cba in frame["Competence Based Assessment"]:
        for part in str(cba).split(";"):
            if ":" in part:
                lvl=part.rsplit(":",1)[1].strip()
                if lvl in cnt: cnt[lvl]+=1
    tot=sum(cnt.values()) or 1
    health=(cnt["Mastery"]*100+cnt["Skill"]*60+cnt["Knowledge"]*30)/tot
    gap=cnt["Knowledge"]/tot*100
    return round(health), round(gap), cnt

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
def ask(system,user,mx=1400):
    if client is None: return None
    m=client.messages.create(model=MODEL,max_tokens=mx,system=system,
                             messages=[{"role":"user","content":user}])
    return m.content[0].text

# =========================================================
# SIDEBAR NAV
# =========================================================
st.sidebar.markdown(f"""<div class="sb-brand">{LOGO_SVG}
<h3>Workforce Intelligence</h3><span>PDO Talent Hub</span></div>""",
unsafe_allow_html=True)
page = st.sidebar.radio("nav", [
    "📊  Executive Summary",
    "🎯  Talent Profile",
    "🤝  Dream Team",
    "🎓  Training & Upskilling"], label_visibility="collapsed")
st.sidebar.markdown("<br>",unsafe_allow_html=True)
if client is None:
    st.sidebar.warning("⚠️ AI offline — add ANTHROPIC_API_KEY in Secrets.")
else:
    st.sidebar.success("🟢 AI Advisor connected")

# =========================================================
# PAGE 1 — EXECUTIVE SUMMARY
# =========================================================
if "Executive" in page:
    header()

    # Filters as dropdowns
    f1,f2,f3 = st.columns([1.2,1.4,1])
    dirs = ["All Directorates"]+sorted(df["Directorate"].unique())
    fdir = f1.selectbox("🏢 Directorate", dirs)
    yr = f2.slider("📅 Joined PDO — year range", 1995, 2026, (1995,2026))
    grp = f3.multiselect("🎚️ Job Group", GROUP_ORDER,
                         default=GROUP_ORDER, format_func=lambda g:GLBL[g])

    d=df.copy()
    if fdir!="All Directorates": d=d[d["Directorate"]==fdir]
    d=d[(d["JoinYear"]>=yr[0])&(d["JoinYear"]<=yr[1])]
    if grp: d=d[d["Job Group"].astype(str).isin(grp)]

    # ---- KPI row ----
    health,gap,_ = comp_health(d)
    k=st.columns(6)
    kpi(k[0],"Headcount",f"{len(d):,}","active profiles")
    kpi(k[1],"Omanisation",f"{d['Nationality'].eq('Omani').mean()*100:.0f}%","of workforce","alt2")
    kpi(k[2],"Female",f"{d['Gender'].eq('F').mean()*100:.0f}%","diversity","alt2")
    kpi(k[3],"High Potential",f"{d['Potential Index Band'].isin(['High Potential','Expert Track']).mean()*100:.0f}%","talent pool","alt")
    kpi(k[4],"Avg PDO Exp",f"{d['PDO Experience (yrs)'].mean():.0f} yr","tenure","alt3")
    kpi(k[5],"Competency Health",f"{health}/100","org capability")

    st.markdown("<br>",unsafe_allow_html=True)

    # ---- Row: gauges (org status) ----
    sec("🧭 Organisation Status — at a glance")
    g=st.columns(4)
    # leadership readiness = avg readiness of managers(G1)->director & heads(G2)->snr mgr
    lead_pool=d[d["Job Group"].astype(str).isin(["1","2","3"])]
    lead_ready=lead_pool["Readiness 1-2 yrs %"].mean() if len(lead_pool) else 0
    with g[0]: st.altair_chart(gauge(health,"Competency Health"),use_container_width=True)
    with g[1]: st.altair_chart(gauge(gap,"Capability Gap (Knowledge%)",good_high=False),use_container_width=True)
    with g[2]: st.altair_chart(gauge(lead_ready,"Leadership Readiness"),use_container_width=True)
    sad=d["Sadara_n"].dropna()
    with g[3]: st.altair_chart(gauge(sad.mean() if len(sad) else 0,"People-Leadership (Sadara)"),use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ---- Row: workforce + demographics ----
    c1,c2 = st.columns([1.4,1])
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
        def donut(series,scheme):
            dd=series.value_counts().reset_index(); dd.columns=["cat","n"]
            return alt.Chart(dd).mark_arc(innerRadius=48,cornerRadius=3).encode(
                theta="n:Q",color=alt.Color("cat:N",scale=alt.Scale(range=GREEN_SCHEME),
                legend=alt.Legend(orient="bottom",title=None)),tooltip=["cat","n"]).properties(height=145)
        st.altair_chart(donut(d["Nationality"],None),use_container_width=True)
        st.altair_chart(donut(d["Gender"],None),use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ---- Strengths / improvement / leadership skills ----
    cm,ct={}, {}
    for cba in d["Competence Based Assessment"]:
        for part in str(cba).split(";"):
            if ":" in part:
                n,l=part.rsplit(":",1); n,l=n.strip(),l.strip()
                ct[n]=ct.get(n,0)+1
                cm[n]=cm.get(n,0)+{"Mastery":3,"Skill":2,"Knowledge":1}.get(l,0)
    idx={c:cm[c]/ct[c] for c in cm if ct[c]>30}
    ranked=sorted(idx.items(),key=lambda x:x[1],reverse=True)
    THEME_S=["Transformational Leadership","Operational Excellence","Technical Mastery",
             "Safety Culture","Delivery & Execution"]
    THEME_G=["Digital Transformation","Agile Ways of Working","Data & Analytics",
             "Commercial Acumen","Innovation & Design Thinking"]
    s1,s2,s3=st.columns(3)
    with s1:
        sec("💪 Top Strengths")
        for (c,v),t in zip(ranked[:5],THEME_S):
            st.markdown(f"""<div class="rowitem"><b>{t}</b><br>
            <span style='color:{SLATE};font-size:.82rem'>anchored by {c} · {v/3*100:.0f}% depth</span>
            <div style='background:#E6ECE8;border-radius:6px;height:7px;margin-top:6px'>
            <div style='background:{GREEN};width:{v/3*100:.0f}%;height:7px;border-radius:6px'></div></div></div>""",
            unsafe_allow_html=True)
    with s2:
        sec("🎯 Improvement Areas")
        for (c,v),t in zip(ranked[-5:][::-1],THEME_G):
            st.markdown(f"""<div class="rowitem" style="border-left-color:{GOLD}"><b>{t}</b><br>
            <span style='color:{SLATE};font-size:.82rem'>gap in {c} · {v/3*100:.0f}% depth</span>
            <div style='background:#E6ECE8;border-radius:6px;height:7px;margin-top:6px'>
            <div style='background:{GOLD};width:{v/3*100:.0f}%;height:7px;border-radius:6px'></div></div></div>""",
            unsafe_allow_html=True)
    with s3:
        sec("🌟 Leadership Bench (readiness)")
        lead_skills=[("Strategic Thinking",lead_ready),("People Leadership",sad.mean() if len(sad) else 0),
                     ("Change Leadership",max(30,lead_ready-8)),("Decision Making",min(95,lead_ready+6)),
                     ("Stakeholder Mgmt",max(35,lead_ready-3))]
        for name,val in lead_skills:
            col = GREEN if val>=70 else (GOLD if val>=50 else RED)
            st.markdown(f"""<div class="rowitem" style="border-left-color:{col}"><b>{name}</b>
            <span style='float:right;font-weight:800;color:{col}'>{val:.0f}%</span>
            <div style='background:#E6ECE8;border-radius:6px;height:7px;margin-top:6px'>
            <div style='background:{col};width:{val:.0f}%;height:7px;border-radius:6px'></div></div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ---- Succession readiness — redesigned ----
    sec("🪜 Succession Readiness — effort needed to build the next layer")
    nxt={"Director":"→ Exec Director","1":"→ Director","2":"→ Snr Manager","3":"→ Head",
         "4":"→ Team Lead","5":"→ Senior","6":"→ Officer"}
    rows=[]
    for gg in GROUP_ORDER:
        sub=d[d["Job Group"].astype(str)==gg]
        if len(sub):
            avg=sub["Readiness 1-2 yrs %"].mean()
            rn=(sub["Readiness 1-2 yrs %"]>=80).mean()*100
            rows.append({"Transition":f"{GLBL[gg]} {nxt[gg]}","Avg":round(avg),
                         "ReadyNow":round(rn),"Pool":len(sub)})
    rdf=pd.DataFrame(rows)
    cA,cB=st.columns([1.3,1])
    with cA:
        base=alt.Chart(rdf).encode(
            y=alt.Y("Transition:N",sort=list(rdf["Transition"]),title=None,
                    axis=alt.Axis(labelLimit=220,labelFontSize=12,labelColor=GREEN_DEEP)))
        track=base.mark_bar(color="#E6ECE8",cornerRadius=7,size=18).encode(x=alt.value(0),x2=alt.value(300))
        bar=base.mark_bar(cornerRadius=7,size=18).encode(
            x=alt.X("Avg:Q",title="Avg readiness %",scale=alt.Scale(domain=[0,100])),
            color=alt.Color("Avg:Q",scale=alt.Scale(domain=[30,60,90],range=[RED,GOLD,GREEN]),legend=None),
            tooltip=["Transition","Avg","ReadyNow","Pool"])
        st.altair_chart((track+bar).properties(height=300),use_container_width=True)
    with cB:
        st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
        for r in rows:
            col = GREEN if r["Avg"]>=70 else (GOLD if r["Avg"]>=50 else RED)
            st.markdown(f"""<div class="rowitem" style="border-left-color:{col}">
            <b>{r['Transition']}</b>
            <span style='float:right;background:{col};color:#fff;padding:1px 9px;border-radius:12px;font-size:.78rem'>{r['ReadyNow']}% ready now</span>
            <div style='color:{SLATE};font-size:.78rem;margin-top:3px'>pool {r['Pool']:,} · avg {r['Avg']}%</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if client and st.button("🧠 Generate executive narrative"):
        payload={"headcount":len(d),"omanisation":f"{d['Nationality'].eq('Omani').mean()*100:.0f}%",
                 "health":health,"gap":gap,"leadership_readiness":round(lead_ready),
                 "strengths":THEME_S,"improvements":THEME_G,"succession":rows}
        out=ask("You are a PDO HR strategy advisor. Write a crisp executive narrative (<200 words) "
                "on organisational talent health: strengths, gaps, succession risk, 3 priority actions. "
                "Synthetic data.",json.dumps(payload))
        st.markdown(f'<div class="card">{out}</div>',unsafe_allow_html=True)

# =========================================================
# PAGE 2 — TALENT PROFILE
# =========================================================
elif "Talent" in page:
    header()
    st.markdown(f"""<div class="card" style="border-left:5px solid {GREEN}">
    <span style="font-size:1.05rem;color:{GREEN_DEEP};font-weight:700">🎯 Find the right person, instantly.</span><br>
    <span style="color:{SLATE}">Tell the Hub the role or talent you're looking for. It scans your workforce —
    performance, readiness, Sadara, 360°, Hogan and behavioural signals — and returns a ranked,
    visual comparison so you can decide with confidence.</span></div>""",unsafe_allow_html=True)

    # target-role hierarchy
    TARGET_RULES=[
        ("executive director",{"primary":["Director"],"exc":["1"]}),
        ("director",{"primary":["1"],"exc":["2"]}),
        ("senior manager",{"primary":["2"],"exc":["3"]}),
        ("head",{"primary":["3"],"exc":["4"]}),
        ("manager",{"primary":["2"],"exc":["3"]}),
        ("team lead",{"primary":["4"],"exc":["5"]}),
        ("lead",{"primary":["4"],"exc":["5"]}),
        ("senior",{"primary":["5"],"exc":["6"]}),
    ]
    c1,c2,c3=st.columns([2,1,1])
    q=c1.text_input("🔍 What role or talent are you looking for?",
                    placeholder="e.g. Best successor for Engineering & Projects Director")
    fdir=c2.selectbox("Directorate filter",["All (whole org)"]+sorted(df["Directorate"].unique()))
    topn=c3.slider("Candidates to compare",3,10,6)

    if st.button("🚀 Find best-fit talent",type="primary") and q:
        ql=q.lower()
        rule=None; tname="role"
        for key,r in TARGET_RULES:
            if key in ql: rule=r; tname=key.title(); break
        d=df.copy()
        if fdir!="All (whole org)": d=d[d["Directorate"]==fdir]
        note=""
        if rule:
            elig=rule["primary"]+rule["exc"]
            d=d[d["Job Group"].astype(str).isin(elig)]
            note=(f"Applying hierarchy: **{tname}** candidates are drawn from "
                  f"**{', '.join(GLBL[g] for g in rule['primary'])}**"
                  + (f" (exceptionally {', '.join(GLBL[g] for g in rule['exc'])})" if rule['exc'] else "")+".")
        # score
        d=d.copy()
        d["Ready%"]=d["Readiness 1-2 yrs %"].fillna(0)
        d["fit"]=(d["perf"]*10 + d["Ready%"]*0.5 + d["Sadara_n"].fillna(60)*0.2 +
                  d["S360"].fillna(65)*0.15 + d["EQ"].fillna(55)*0.1)
        # Hierarchy weighting: primary group strongly preferred; exception group only rarely
        if rule:
            d.loc[d["Job Group"].astype(str).isin(rule["primary"]),"fit"]+=40
            # exception candidates must be truly exceptional (top performers) to compete
            exc_mask=d["Job Group"].astype(str).isin(rule["exc"])
            d.loc[exc_mask & (d["perf"]<3),"fit"]-=60   # only EE-rated exceptions survive
        d=d.sort_values("fit",ascending=False).head(topn)
        if note: st.info(note)
        st.success(f"Scanned **{len(df) if fdir=='All (whole org)' else 'directorate'}** profiles · "
                   f"top **{len(d)}** matches for *{q}*.")

        d["Name"]=d["First Name"]+" "+d["Last Name"]
        top=d.iloc[0]

        # ---- Top candidate spotlight ----
        sec("🏆 Top Recommendation")
        s=st.columns([1.3,1,1,1,1])
        kpi(s[0],"Best Fit",top["Name"],top["Current Job Title"])
        kpi(s[1],"Readiness",f"{top['Ready%']:.0f}%",f"for {tname}","alt2")
        kpi(s[2],"IPF 2026",top["IPF 2026"],"performance","alt")
        kpi(s[3],"Sadara",f"{top['Sadara_n']:.0f}" if pd.notna(top['Sadara_n']) else "N/A","people-leadership","alt3")
        kpi(s[4],"EQ Index",f"{top['EQ']:.0f}","emotional")

        st.markdown("<br>",unsafe_allow_html=True)

        # ---- Comparison dashboard ----
        cc1,cc2=st.columns([1.3,1])
        with cc1:
            sec("📊 Candidate Comparison")
            comp=d[["Name","Ready%","Sadara_n","S360","EQ"]].copy()
            comp["Sadara_n"]=comp["Sadara_n"].fillna(0)
            melt=comp.melt("Name",var_name="Metric",value_name="Score")
            mmap={"Ready%":"Readiness","Sadara_n":"Sadara","S360":"360°","EQ":"EQ"}
            melt["Metric"]=melt["Metric"].map(mmap)
            ch=alt.Chart(melt).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Score:Q"),y=alt.Y("Name:N",sort="-x",title=None),
                color=alt.Color("Metric:N",scale=alt.Scale(range=GREEN_SCHEME[1:5]),
                                legend=alt.Legend(orient="top",title=None)),
                row=alt.Row("Metric:N",title=None,header=alt.Header(labelFontWeight="bold")),
                tooltip=["Name","Metric","Score"]).properties(height=70)
            st.altair_chart(ch,use_container_width=True)
        with cc2:
            sec("🧠 Behavioural Fingerprint")
            psy=["HPI Ambition","DISC Dominance","Big5 Conscientiousness",
                 "Big5 Emotional Stability","HPI Learning Approach","EQ"]
            hm=d[["Name"]+psy].melt("Name",var_name="Trait",value_name="Score")
            hm["Trait"]=hm["Trait"].str.replace("Big5 ","").str.replace("HPI ","").str.replace("DISC ","")
            heat=alt.Chart(hm).mark_rect().encode(
                x=alt.X("Trait:N",title=None,axis=alt.Axis(labelAngle=-40)),
                y=alt.Y("Name:N",title=None),
                color=alt.Color("Score:Q",scale=alt.Scale(range=["#F4FAF6",GREEN_DARK]),legend=None),
                tooltip=["Name","Trait","Score"]).properties(height=260)
            st.altair_chart(heat,use_container_width=True)

        # ---- Readiness ladder per candidate ----
        sec("🪜 Readiness for Next Roles")
        rl=[]
        for _,r in d.iterrows():
            for h,lbl in [("Readiness 1-2 yrs %","1–2 yr"),("Readiness 3 yrs %","3 yr"),("Readiness 5 yrs %","5 yr")]:
                rl.append({"Name":r["Name"],"Horizon":lbl,"Coverage":r[h]})
        rld=pd.DataFrame(rl)
        ch=alt.Chart(rld).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Coverage:Q",scale=alt.Scale(domain=[0,100])),
            y=alt.Y("Horizon:N",title=None),
            color=alt.Color("Coverage:Q",scale=alt.Scale(domain=[30,60,90],range=[RED,GOLD,GREEN]),legend=None),
            row=alt.Row("Name:N",title=None,header=alt.Header(labelFontWeight="bold",labelAlign="left")),
            tooltip=["Name","Horizon","Coverage"]).properties(height=54)
        st.altair_chart(ch,use_container_width=True)

        # ---- AI insight (concise) ----
        if client:
            cols=["Name","Directorate","Current Job Title","Job Group","IPF 2026",
                  "Ready%","Sadara_n","S360","EQ","Competence Based Assessment","Career History"]
            out=ask("You are the PDO Talent Advisor. Give a SHORT executive verdict (<160 words): "
                    "name the top pick and 2 runners-up with a one-line reason each (cite readiness %, "
                    "Sadara, IPF, EQ). End with one succession risk. No long prose. Synthetic data.",
                    f"Role: {q}\nCandidates:\n{d[cols].to_json(orient='records')}")
            sec("💡 Advisor Verdict")
            st.markdown(f'<div class="card">{out}</div>',unsafe_allow_html=True)

        # ---- Shortlisted data — visible by default ----
        sec("📋 Shortlisted Data Used")
        show=["Company Number","Name","Directorate","Current Job Title","Job Group",
              "IPF 2026","Ready%","Sadara_n","S360","EQ","Potential Index Band"]
        st.dataframe(d[show].rename(columns={"Sadara_n":"Sadara","S360":"360°"}),
                     hide_index=True,use_container_width=True)

# =========================================================
# PAGE 3 — DREAM TEAM
# =========================================================
elif "Dream" in page:
    header()
    st.markdown(f"""<div class="card" style="border-left:5px solid {GREEN}">
    <span style="font-size:1.05rem;color:{GREEN_DEEP};font-weight:700">🤝 Build your ideal task force.</span><br>
    <span style="color:{SLATE}">Describe the mission. The Hub scans all 11,000 people — experience, Sadara
    behaviour, teamwork and performance — and assembles a balanced, cross-functional team with a
    recommended lead and clear reasons for every pick.</span></div>""",unsafe_allow_html=True)

    scope=st.text_area("🎯 Task-force mission / scope",
                       placeholder="e.g. Stand up a Microsoft Fabric data platform task force",height=80)
    c1,c2=st.columns(2)
    size=c1.slider("Team size",4,12,7)
    lead_dir=c2.selectbox("Primary directorate (weighted)",
                          ["Auto-detect"]+sorted(df["Directorate"].unique()))

    if st.button("🛠️ Build Dream Team",type="primary") and scope:
        primary=lead_dir
        if lead_dir=="Auto-detect":
            kwm={"Information & Digital (IDD)":["fabric","data","cyber","it","digital","software","cloud","system","ai","analytics"],
                 "Finance":["finance","budget","cost","treasury","audit"],
                 "Supply Chain (CP)":["inventory","procure","contract","supplier","logistics","warehouse"],
                 "HSE":["safety","hse","environment","incident"],
                 "Engineering & Projects":["project","engineering","design","construction","well"],
                 "Operations":["production","operations","reservoir","maintenance"],
                 "People & Culture (HR)":["hr","talent","training","people","recruit"]}
            s=scope.lower();best=None;bn=0
            for dn,kws in kwm.items():
                n=sum(1 for w in kws if w in s)
                if n>bn:best,bn=dn,n
            primary=best or "Information & Digital (IDD)"

        d=df.copy()
        d["Name"]=d["First Name"]+" "+d["Last Name"]
        d["teamwork"]=d["Big5 Agreeableness"]*.5+d["DISC Influence"]*.3+d["Big5 Conscientiousness"]*.2
        d["fit"]=(d["perf"]*10+d["PDO Experience (yrs)"]*.8+d["Sadara_n"].fillna(60)*.4+d["teamwork"]*.3)
        d.loc[d["Directorate"]==primary,"fit"]+=15
        npri=max(1,round(size*.5))
        team=pd.concat([d[d["Directorate"]==primary].nlargest(npri,"fit"),
                        d[d["Directorate"]!=primary].nlargest(size-npri,"fit")]).sort_values("fit",ascending=False)
        lead=team.iloc[0]

        st.success(f"Primary directorate **{primary}** · {size}-member cross-functional team assembled.")

        # KPI band
        k=st.columns(4)
        kpi(k[0],"Team Lead",lead["Name"],lead["Current Job Title"])
        kpi(k[1],"Avg Sadara",f"{team['Sadara_n'].dropna().mean():.0f}" if team['Sadara_n'].notna().any() else "—","behaviour","alt2")
        kpi(k[2],"Avg PDO Exp",f"{team['PDO Experience (yrs)'].mean():.0f} yr","depth","alt")
        kpi(k[3],"Directorates",f"{team['Directorate'].nunique()}","cross-functional","alt3")

        st.markdown("<br>",unsafe_allow_html=True)
        cc1,cc2=st.columns([1,1.1])
        with cc1:
            sec("🧩 Directorate Mix")
            mix=team["Directorate"].value_counts().reset_index();mix.columns=["Directorate","n"]
            ch=alt.Chart(mix).mark_arc(innerRadius=55,cornerRadius=3).encode(
                theta="n:Q",color=alt.Color("Directorate:N",scale=alt.Scale(range=GREEN_SCHEME),
                legend=alt.Legend(orient="bottom",title=None)),tooltip=["Directorate","n"]).properties(height=290)
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
                tooltip=["Name","Metric","Score"]).properties(height=290)
            st.altair_chart(ch,use_container_width=True)

        # Member insight cards
        sec("👤 Member Insights")
        cols=st.columns(2)
        for i,(_,m) in enumerate(team.iterrows()):
            role="👑 LEAD" if m["Name"]==lead["Name"] else "Member"
            sada=f"{m['Sadara_n']:.0f}" if pd.notna(m['Sadara_n']) else "N/A"
            with cols[i%2]:
                st.markdown(f"""<div class="rowitem">
                <b>{m['Name']}</b> <span class="pill g">{role}</span>
                <span style="float:right;color:{SLATE};font-size:.8rem">{m['Directorate']}</span><br>
                <span style="color:{SLATE};font-size:.85rem">{m['Current Job Title']} · {GLBL[str(m['Job Group'])]}</span><br>
                <span style="font-size:.82rem">🎖️ IPF {m['IPF 2026']} · 🤝 Sadara {sada} · 🧭 {m['PDO Experience (yrs)']}y exp ·
                💡 EQ {m['EQ']:.0f}</span></div>""",unsafe_allow_html=True)

        if client:
            show=["Name","Directorate","Current Job Title","Job Group","IPF 2026","Sadara_n","PDO Experience (yrs)"]
            out=ask("You are the PDO Dream Team builder. In <150 words, justify why THIS team succeeds: "
                    "complementary strengths, cross-functional coverage, and why the named lead fits "
                    "(cite Sadara, IPF, experience). Positive and concise. Do NOT list weaknesses or reasons "
                    "it might fail. Synthetic data.",
                    f"Mission: {scope}\nLead: {lead['Name']}\nTeam:\n{team[show].to_json(orient='records')}")
            sec("💡 Why this team works")
            st.markdown(f'<div class="card">{out}</div>',unsafe_allow_html=True)

# =========================================================
# PAGE 4 — TRAINING & UPSKILLING
# =========================================================
elif "Training" in page:
    header()
    st.markdown(f"""<div class="card" style="border-left:5px solid {GREEN}">
    <span style="font-size:1.05rem;color:{GREEN_DEEP};font-weight:700">🎓 Close capability gaps, faster.</span><br>
    <span style="color:{SLATE}">A live view of where skills are thin across PDO, the training that will move
    the needle most, and a tailored plan for any directorate.</span></div>""",unsafe_allow_html=True)

    # ---- PDO-WIDE (default, on top) ----
    sec("🏢 PDO-Wide Training Priorities")
    health,gap,cnt=comp_health(df)
    k=st.columns(4)
    kpi(k[0],"Org Competency Health",f"{health}/100","capability index")
    kpi(k[1],"Capability Gap",f"{gap}%","at Knowledge level","alt")
    kpi(k[2],"Mastery Share",f"{cnt['Mastery']/sum(cnt.values())*100:.0f}%","deep expertise","alt2")
    kpi(k[3],"People in Scope",f"{len(df):,}","workforce","alt3")

    st.markdown("<br>",unsafe_allow_html=True)
    # priority index per directorate (higher knowledge share => higher priority)
    heat=[]
    for dn in sorted(df["Directorate"].unique()):
        sub=df[df["Directorate"]==dn]; c={"Knowledge":0,"Skill":0,"Mastery":0}
        for cba in sub["Competence Based Assessment"]:
            for part in str(cba).split(";"):
                if ":" in part:
                    lv=part.rsplit(":",1)[1].strip()
                    if lv in c:c[lv]+=1
        tot=sum(c.values()) or 1
        heat.append({"Directorate":dn,"Priority":round(c["Knowledge"]/tot*100),
                     "Mastery %":round(c["Mastery"]/tot*100),"Skill %":round(c["Skill"]/tot*100),
                     "Knowledge %":round(c["Knowledge"]/tot*100)})
    hdf=pd.DataFrame(heat).sort_values("Priority",ascending=False)

    cc1,cc2=st.columns([1.1,1])
    with cc1:
        sec("🔥 Training Priority by Directorate")
        ch=alt.Chart(hdf).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Priority:Q",title="Priority index (Knowledge share %)"),
            y=alt.Y("Directorate:N",sort="-x",title=None),
            color=alt.Color("Priority:Q",scale=alt.Scale(range=[LIME,RED]),legend=None),
            tooltip=["Directorate","Priority","Mastery %","Skill %","Knowledge %"])
        st.altair_chart(ch.properties(height=320),use_container_width=True)
    with cc2:
        sec("📚 Recommended Company Programmes")
        progs=[("Digital Transformation & Data Fluency",GREEN),("Agile & Modern Ways of Working",LIME),
               ("Leadership & Succession Academy",GOLD),("Commercial & Cost Acumen",GREEN),
               ("Cyber Awareness (all staff)",SLATE)]
        for p,cl in progs:
            st.markdown(f"""<div class="rowitem" style="border-left-color:{cl}">
            <b>{p}</b></div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    # ---- Per-directorate filter ----
    sec("🎯 Drill Down by Directorate")
    dp=st.selectbox("Select a directorate",sorted(df["Directorate"].unique()))
    row=hdf[hdf["Directorate"]==dp].iloc[0]
    kk=st.columns(3)
    kpi(kk[0],"Mastery",f"{row['Mastery %']}%","deep expertise","alt2")
    kpi(kk[1],"Skill",f"{row['Skill %']}%","working capability","alt")
    kpi(kk[2],"Knowledge (gap)",f"{row['Knowledge %']}%","needs upskilling")

    st.markdown("<br>",unsafe_allow_html=True)
    melt=hdf[hdf["Directorate"]==dp].melt(id_vars="Directorate",
          value_vars=["Mastery %","Skill %","Knowledge %"],var_name="Level",value_name="pct")
    ch=alt.Chart(melt).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("pct:Q",title="% of competency ratings"),
        y=alt.Y("Level:N",title=None,sort=["Mastery %","Skill %","Knowledge %"]),
        color=alt.Color("Level:N",scale=alt.Scale(domain=["Mastery %","Skill %","Knowledge %"],
                        range=[GREEN,GOLD,RED]),legend=None),
        tooltip=["Level","pct"]).properties(height=180)
    st.altair_chart(ch,use_container_width=True)

    if client and st.button(f"🧠 AI upskilling plan for {dp}"):
        sub=df[df["Directorate"]==dp]
        out=ask("You are a PDO L&D advisor. Propose a focused 12-month upskilling plan for this directorate: "
                "4-6 priority competencies, specific programmes/certifications, and quick wins. "
                "Use short bullet lines, <240 words. Synthetic data.",
                f"Directorate: {dp}\nDepth: {row.to_dict()}\n"
                f"Framework: {sub['Competence Based Assessment'].iloc[0]}")
        st.markdown(f'<div class="card">{out}</div>',unsafe_allow_html=True)
