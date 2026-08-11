"""
PDO Talent Intelligence Platform
Pages: Executive Summary | Talent Profile (chat) | Dream Team | Training & Upskilling
Phase 1: CSV-powered. Swap load_data() for Supabase later. ALL DATA SYNTHETIC.
"""
import os, json, re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="PDO Talent Intelligence", page_icon="🧭", layout="wide")

DATA_FILE = "pdo_talent_profiles.csv"
MODEL = "claude-sonnet-4-5-20250929"

# ---------------- Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, keep_default_na=False)
    df["JoinYear"] = df["Date Joined PDO"].str[:4].astype(int)
    for h in ["Readiness 1-2 yrs", "Readiness 3 yrs", "Readiness 5 yrs"]:
        df[h + " %"] = df[h].apply(lambda s: int(re.search(r"(\d+)%", str(s)).group(1))
                                   if re.search(r"(\d+)%", str(s)) else np.nan)
    return df

df = load_data()
GROUP_ORDER = ["Director", "1", "2", "3", "4", "5", "6"]
GROUP_LABEL = {"Director":"Director","1":"G1 Manager","2":"G2 Head","3":"G3 Lead",
               "4":"G4 Senior","5":"G5 Junior","6":"G6 Graduate"}

# ---------------- Claude ----------------
def get_client():
    try:
        from anthropic import Anthropic
    except Exception:
        return None
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    return Anthropic(api_key=key) if key else None

client = get_client()

def ask_claude(system, user, max_tokens=1600):
    if client is None:
        return None
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                 system=system, messages=[{"role":"user","content":user}])
    return msg.content[0].text

# ---------------- Sidebar nav ----------------
st.sidebar.title("🧭 PDO Talent Intelligence")
page = st.sidebar.radio("Navigate", [
    "📊 Executive Summary", "💬 Talent Profile", "🤝 Dream Team", "🎓 Training & Upskilling"])
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} synthetic profiles · Phase 1 (CSV)")
if client is None:
    st.sidebar.warning("AI offline — add ANTHROPIC_API_KEY in Secrets to enable chat pages.")

# ==========================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ==========================================================
if page == "📊 Executive Summary":
    st.title("📊 Executive Summary — Organisation Health")

    # Interactive filters
    c1, c2, c3 = st.columns(3)
    dsel = c1.multiselect("Directorate", sorted(df["Directorate"].unique()),
                          default=sorted(df["Directorate"].unique()))
    yr = c2.slider("Joined PDO on/before", int(df["JoinYear"].min()),
                   int(df["JoinYear"].max()), int(df["JoinYear"].max()))
    gsel = c3.multiselect("Job Group", GROUP_ORDER, default=GROUP_ORDER)
    d = df[df["Directorate"].isin(dsel) & (df["JoinYear"] <= yr) &
           df["Job Group"].astype(str).isin(gsel)]
    st.caption(f"Showing **{len(d):,}** employees based on filters.")

    # KPI row
    k = st.columns(5)
    k[0].metric("Headcount", f"{len(d):,}")
    k[1].metric("Omanisation", f"{d['Nationality'].eq('Omani').mean()*100:.1f}%")
    k[2].metric("Female %", f"{d['Gender'].eq('F').mean()*100:.1f}%")
    k[3].metric("Avg PDO Exp", f"{d['PDO Experience (yrs)'].mean():.1f} yrs")
    hi = d["Potential Index Band"].isin(["High Potential","Expert Track"]).mean()*100
    k[4].metric("High Potential", f"{hi:.0f}%")

    st.markdown("---")
    cL, cR = st.columns(2)

    # Group pyramid
    with cL:
        st.subheader("👥 Workforce by Job Group")
        vc = d["Job Group"].astype(str).value_counts().reindex(GROUP_ORDER).fillna(0)
        pdata = pd.DataFrame({"Group": [GROUP_LABEL[g] for g in GROUP_ORDER],
                              "Count": vc.values.astype(int)})
        chart = alt.Chart(pdata).mark_bar().encode(
            x=alt.X("Count:Q", title=""),
            y=alt.Y("Group:N", sort=[GROUP_LABEL[g] for g in GROUP_ORDER], title=""),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Group","Count"]).properties(height=330)
        st.altair_chart(chart, use_container_width=True)

    # Gender & nationality donut
    with cR:
        st.subheader("🌍 Nationality & Gender")
        g1, g2 = st.columns(2)
        def donut(series, title, scheme):
            dd = series.value_counts().reset_index()
            dd.columns = ["cat", "n"]
            ch = alt.Chart(dd).mark_arc(innerRadius=55).encode(
                theta="n:Q", color=alt.Color("cat:N", scale=alt.Scale(scheme=scheme),
                                              legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["cat","n"]).properties(height=270, title=title)
            return ch
        g1.altair_chart(donut(d["Nationality"], "Nationality", "blues"), use_container_width=True)
        g2.altair_chart(donut(d["Gender"], "Gender", "teals"), use_container_width=True)

    st.markdown("---")

    # ---- Top strengths & improvement areas (derived from competency mastery) ----
    st.subheader("💪 Top Organisational Strengths & 🎯 Improvement Areas")
    # Build competency mastery index across org
    comp_master = {}
    comp_total = {}
    for cba in d["Competence Based Assessment"]:
        for part in str(cba).split(";"):
            if ":" in part:
                name, lvl = part.rsplit(":", 1)
                name, lvl = name.strip(), lvl.strip()
                comp_total[name] = comp_total.get(name, 0) + 1
                score = {"Mastery":3, "Skill":2, "Knowledge":1}.get(lvl, 0)
                comp_master[name] = comp_master.get(name, 0) + score
    idx = {c: comp_master[c]/comp_total[c] for c in comp_master if comp_total[c] > 30}
    ranked = sorted(idx.items(), key=lambda x: x[1], reverse=True)
    strengths = ranked[:5]
    gaps = ranked[-5:][::-1]
    # Map generic strengths/areas to leadership themes
    THEME_STRONG = ["Transformational Leadership", "Operational Excellence", "Technical Depth",
                    "Safety Culture", "Execution & Delivery"]
    THEME_GAP = ["Digital Transformation", "Agile Ways of Working", "Data & Analytics",
                 "Commercial Acumen", "Innovation & Design Thinking"]
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### 💪 Top 5 Strengths")
        for (c, v), theme in zip(strengths, THEME_STRONG):
            st.markdown(f"- **{theme}** — anchored by strong *{c}* (index {v:.2f}/3)")
    with s2:
        st.markdown("#### 🎯 Top 5 Improvement Areas")
        for (c, v), theme in zip(gaps, THEME_GAP):
            st.markdown(f"- **{theme}** — weak coverage in *{c}* (index {v:.2f}/3)")

    st.markdown("---")

    # ---- Succession readiness per group (bridge to next level) ----
    st.subheader("🪜 Succession Readiness — % Ready for the Next Level")
    st.caption("Average competency coverage for the next role (1–2 yr horizon), by current group.")
    bridge = []
    nextlbl = {"1":"→ Director","2":"→ Senior Manager","3":"→ Head",
               "4":"→ Team Lead","5":"→ Senior","6":"→ Officer","Director":"→ Exec Director"}
    for g in GROUP_ORDER:
        sub = d[d["Job Group"].astype(str) == g]
        if len(sub):
            bridge.append({"Group": f"{GROUP_LABEL[g]} {nextlbl[g]}",
                           "Avg Readiness %": round(sub["Readiness 1-2 yrs %"].mean(), 0),
                           "Ready Now %": round((sub["Readiness 1-2 yrs %"] >= 80).mean()*100, 0)})
    bdf = pd.DataFrame(bridge)
    chart = alt.Chart(bdf).mark_bar(color="#2E86C1").encode(
        x=alt.X("Avg Readiness %:Q", title="% competency coverage for next role"),
        y=alt.Y("Group:N", sort=list(bdf["Group"]), title=""),
        tooltip=list(bdf.columns)).properties(height=340)
    text = chart.mark_text(align="left", dx=3).encode(text="Avg Readiness %:Q")
    st.altair_chart(chart + text, use_container_width=True)
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- Extra insight: IPF performance & Sadara health ----
    e1, e2 = st.columns(2)
    with e1:
        st.subheader("📈 Performance (IPF 2026)")
        ipf = d[d["IPF 2026"] != ""]["IPF 2026"].value_counts().reindex(["EE","AE","MM"]).fillna(0)
        idata = pd.DataFrame({"Rating":["Exceed (EE)","Above (AE)","Meet (MM)"],
                              "Count": ipf.values.astype(int),
                              "c":["EE","AE","MM"]})
        ch = alt.Chart(idata).mark_bar().encode(
            x=alt.X("Rating:N", sort=None, title=""),
            y=alt.Y("Count:Q", title=""),
            color=alt.Color("c:N", scale=alt.Scale(domain=["EE","AE","MM"],
                            range=["#27AE60","#F39C12","#5D6D7E"]), legend=None),
            tooltip=["Rating","Count"]).properties(height=300)
        st.altair_chart(ch, use_container_width=True)
    with e2:
        st.subheader("🧑‍🤝‍🧑 Sadara People-Leadership Health")
        sada = pd.to_numeric(d["Sadara Survey (people leadership)"], errors="coerce").dropna()
        avg = sada.mean() if len(sada) else 0
        color = "#E74C3C" if avg < 60 else ("#F39C12" if avg < 80 else "#27AE60")
        st.metric("Avg Sadara (Lead & above)", f"{avg:.1f} / 100")
        st.progress(min(1.0, avg/100))
        st.caption(f"🔴 40–59 · 🟠 60–79 · 🟢 80–100 — current band: "
                   f"{'🔴 Red' if avg<60 else ('🟠 Orange' if avg<80 else '🟢 Green')}")
        # distribution (pre-binned with numpy -> tiny dataframe)
        if len(sada):
            counts, edges = np.histogram(sada, bins=np.arange(40, 102, 3))
            sdata = pd.DataFrame({"Score": [f"{int(edges[i])}-{int(edges[i+1])}"
                                            for i in range(len(counts))],
                                  "People": counts})
            ch = alt.Chart(sdata).mark_bar(color=color).encode(
                x=alt.X("Score:N", sort=None, title="Sadara score band"),
                y=alt.Y("People:Q", title="People"),
                tooltip=["Score","People"]).properties(height=200)
            st.altair_chart(ch, use_container_width=True)

    # ---- AI narrative (optional) ----
    st.markdown("---")
    if st.button("🧠 Generate AI executive narrative"):
        summary = {
            "headcount": len(d), "omanisation": f"{d['Nationality'].eq('Omani').mean()*100:.1f}%",
            "female": f"{d['Gender'].eq('F').mean()*100:.1f}%",
            "strengths": THEME_STRONG, "improvement": THEME_GAP,
            "succession": bridge}
        out = ask_claude(
            "You are a PDO HR strategy advisor. Write a crisp executive narrative "
            "(<220 words) on organisational talent health: strengths, gaps, succession "
            "risk and 3 priority actions. Data is synthetic.",
            json.dumps(summary))
        st.markdown(out or "_Add ANTHROPIC_API_KEY to enable AI narrative._")

# ==========================================================
# PAGE 2 — TALENT PROFILE (CHAT)
# ==========================================================
elif page == "💬 Talent Profile":
    st.title("💬 Talent Profile — Best-Fit Advisor")
    st.caption("Ask who fits a role; the app filters candidates and Claude compares them.")

    with st.sidebar:
        st.subheader("🔎 Pre-filter")
        f_dir = st.selectbox("Directorate", ["All"] + sorted(df["Directorate"].unique()))
        f_grp = st.multiselect("Job Group", GROUP_ORDER, default=["Director","1","2","3"])
        f_pot = st.selectbox("Potential", ["All"] + sorted(df["Potential Index Band"].unique()))
        kw = st.text_input("Keyword (skill/cert/degree)")
        topk = st.slider("Shortlist to AI", 5, 40, 20)

    q = st.text_area("Your question",
                     placeholder="Best successor for Engineering & Projects Director with strong leadership...",
                     height=90)
    if st.button("🚀 Get recommendation", type="primary"):
        d = df.copy()
        if f_dir != "All": d = d[d["Directorate"] == f_dir]
        if f_grp: d = d[d["Job Group"].astype(str).isin(f_grp)]
        if f_pot != "All": d = d[d["Potential Index Band"] == f_pot]
        if kw:
            d = d[d.apply(lambda r: kw.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        ipf_rank = {"EE":3,"AE":2,"MM":1,"":0}
        d = d.copy()
        d["_s"] = (d["IPF 2026"].map(ipf_rank).fillna(0)*10 +
                   d["Readiness 1-2 yrs %"].fillna(0)*0.5 +
                   pd.to_numeric(d["Sadara Survey (people leadership)"], errors="coerce").fillna(60)*0.2)
        d = d.sort_values("_s", ascending=False).head(topk)
        st.info(f"Shortlisted **{len(d)}** candidates → sent to Advisor.")
        cols = ["Company Number","First Name","Last Name","Directorate","Current Job Title",
                "Job Group","IPF 2026","Potential Index Band","Sadara Survey (people leadership)",
                "Readiness 1-2 yrs","Competence Based Assessment","Career History"]
        if client is None:
            st.warning("AI offline — showing ranked shortlist:")
            st.dataframe(d[cols], hide_index=True)
        else:
            with st.spinner("Comparing candidates..."):
                out = ask_claude(
                    "You are the PDO Talent Advisor. Compare the JSON candidates for the role/"
                    "criteria. Rank top 3-5 with **Name, Group, Directorate**, 'Why they fit' "
                    "bullets (cite IPF, readiness %, Sadara, competencies, experience) and a "
                    "'Watch-outs' line. Executive tone. Data is synthetic.",
                    f"Role/criteria: {q}\n\nCandidates:\n{d[cols].to_json(orient='records')}")
                st.markdown(out)
            with st.expander("Shortlist data used"):
                st.dataframe(d[cols], hide_index=True)

# ==========================================================
# PAGE 3 — DREAM TEAM
# ==========================================================
elif page == "🤝 Dream Team":
    st.title("🤝 Dream Team — Task Force Builder")
    st.caption("Describe the mission; we assemble a cross-functional team and name a lead.")

    scope = st.text_area("Task force scope / activity",
                         placeholder="e.g. Build a Microsoft Fabric data platform task force...",
                         height=80)
    c1, c2 = st.columns(2)
    size = c1.slider("Team size", 4, 12, 7)
    lead_dir = c2.selectbox("Primary directorate (gets more weight)",
                            ["Auto-detect"] + sorted(df["Directorate"].unique()))

    if st.button("🛠️ Build the Dream Team", type="primary"):
        # Auto-detect primary directorate from keywords
        primary = lead_dir
        if lead_dir == "Auto-detect":
            kw_map = {"Information & Digital (IDD)":["fabric","data","cyber","it","digital","software","cloud","system"],
                      "Finance":["finance","budget","cost","treasury","audit"],
                      "Supply Chain (CP)":["inventory","procure","contract","supplier","logistics","warehouse"],
                      "HSE":["safety","hse","environment","incident"],
                      "Engineering & Projects":["project","engineering","design","construction"],
                      "People & Culture (HR)":["hr","talent","training","people","recruit"]}
            s = scope.lower(); best=None; bestn=0
            for dname, kws in kw_map.items():
                n = sum(1 for w in kws if w in s)
                if n > bestn: best, bestn = dname, n
            primary = best or "Information & Digital (IDD)"
        st.info(f"Primary directorate: **{primary}** · assembling {size} members across the org.")

        # Score candidates: teamwork (agreeableness), Sadara, experience, performance
        d = df.copy()
        d["Sadara_n"] = pd.to_numeric(d["Sadara Survey (people leadership)"], errors="coerce").fillna(60)
        ipf_rank = {"EE":3,"AE":2,"MM":1,"":0}
        d["perf"] = d["IPF 2026"].map(ipf_rank).fillna(0)
        d["teamwork"] = d["Big5 Agreeableness"]*0.5 + d["DISC Influence"]*0.3 + d["Big5 Conscientiousness"]*0.2
        d["fit"] = (d["perf"]*10 + d["PDO Experience (yrs)"]*0.8 + d["Sadara_n"]*0.4 +
                    d["teamwork"]*0.3)
        d.loc[d["Directorate"] == primary, "fit"] += 15  # weight primary directorate

        # ~50% from primary, rest mixed
        n_primary = max(1, round(size*0.5))
        prim = d[d["Directorate"] == primary].sort_values("fit", ascending=False).head(n_primary)
        others = d[d["Directorate"] != primary].sort_values("fit", ascending=False).head(size - n_primary)
        team = pd.concat([prim, others]).sort_values("fit", ascending=False)
        lead = team.iloc[0]

        show = ["First Name","Last Name","Directorate","Current Job Title","Job Group",
                "IPF 2026","Sadara Survey (people leadership)","PDO Experience (yrs)"]
        st.subheader("👑 Suggested Lead")
        st.success(f"**{lead['First Name']} {lead['Last Name']}** — {lead['Current Job Title']} "
                   f"({lead['Directorate']}) · IPF {lead['IPF 2026']} · Sadara "
                   f"{lead['Sadara Survey (people leadership)']}")
        st.subheader(f"👥 Team ({len(team)} members)")
        st.dataframe(team[show], hide_index=True, use_container_width=True)

        # Directorate mix chart
        mix = team["Directorate"].value_counts().reset_index()
        mix.columns = ["Directorate", "n"]
        ch = alt.Chart(mix).mark_arc(innerRadius=50).encode(
            theta="n:Q", color=alt.Color("Directorate:N",
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Directorate","n"]).properties(height=320, title="Directorate mix")
        st.altair_chart(ch, use_container_width=True)

        if client is not None:
            with st.spinner("Writing justification..."):
                out = ask_claude(
                    "You are the PDO Dream Team builder. Given the mission and the selected team "
                    "(JSON), justify in an executive tone why THIS team will succeed: complementary "
                    "skills, cross-functional coverage, why the named lead fits (cite Sadara, IPF, "
                    "experience), and one risk to manage. <200 words. Data is synthetic.",
                    f"Mission: {scope}\nLead: {lead['First Name']} {lead['Last Name']}\n"
                    f"Team:\n{team[show].to_json(orient='records')}")
                st.markdown("### 🧠 Why this team works")
                st.markdown(out)

# ==========================================================
# PAGE 4 — TRAINING & UPSKILLING
# ==========================================================
elif page == "🎓 Training & Upskilling":
    st.title("🎓 Training & Upskilling — Gap-Driven Plan")
    st.caption("Where competency gaps sit, per directorate, plus recommended interventions.")

    # Gap heat: for each directorate, count Knowledge-level (weakest) competencies
    heat = []
    for dname in sorted(df["Directorate"].unique()):
        sub = df[df["Directorate"] == dname]
        counts = {"Knowledge":0, "Skill":0, "Mastery":0}
        for cba in sub["Competence Based Assessment"]:
            for part in str(cba).split(";"):
                if ":" in part:
                    lvl = part.rsplit(":", 1)[1].strip()
                    if lvl in counts: counts[lvl] += 1
        tot = sum(counts.values()) or 1
        heat.append({"Directorate": dname,
                     "Knowledge %": round(counts["Knowledge"]/tot*100),
                     "Skill %": round(counts["Skill"]/tot*100),
                     "Mastery %": round(counts["Mastery"]/tot*100)})
    hdf = pd.DataFrame(heat).sort_values("Knowledge %", ascending=False)

    st.subheader("🔥 Competency depth by directorate")
    melt = hdf.melt(id_vars="Directorate", value_vars=["Mastery %","Skill %","Knowledge %"],
                    var_name="Level", value_name="pct")
    ch = alt.Chart(melt).mark_bar().encode(
        y=alt.Y("Directorate:N", sort=list(hdf["Directorate"]), title=""),
        x=alt.X("pct:Q", stack="normalize", title="% of competency ratings"),
        color=alt.Color("Level:N", scale=alt.Scale(
            domain=["Mastery %","Skill %","Knowledge %"],
            range=["#27AE60","#F39C12","#E74C3C"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Directorate","Level","pct"]).properties(height=380)
    st.altair_chart(ch, use_container_width=True)
    st.caption("More red (Knowledge) = shallower depth = higher training priority.")

    st.markdown("---")
    st.subheader("🏢 Company-wide training focus")
    focus = ["Digital Transformation & Data Fluency", "Agile & Modern Ways of Working",
             "Leadership & Succession (bench for Director/Manager)",
             "Commercial & Cost Acumen", "Cyber Awareness"]
    for f in focus: st.markdown(f"- **{f}**")

    st.markdown("---")
    st.subheader("🎯 Per-directorate recommendations")
    dpick = st.selectbox("Select directorate", sorted(df["Directorate"].unique()))
    row = hdf[hdf["Directorate"] == dpick].iloc[0]
    st.write(f"**{dpick}** — Mastery {row['Mastery %']}% · Skill {row['Skill %']}% · "
             f"Knowledge {row['Knowledge %']}% (priority ∝ Knowledge).")
    if client is not None and st.button("🧠 AI training plan for this directorate"):
        sub = df[df["Directorate"] == dpick]
        gaps = sub["Competence Based Assessment"].head(50).tolist()
        with st.spinner("Building plan..."):
            out = ask_claude(
                "You are a PDO L&D advisor. Based on the directorate's competency framework and "
                "gap profile, propose a focused 12-month upskilling plan: 4-6 priority competencies, "
                "specific programmes/certifications, and quick wins. Executive tone, <260 words. "
                "Data is synthetic.",
                f"Directorate: {dpick}\nDepth: {row.to_dict()}\nSample assessments: {gaps[:15]}")
            st.markdown(out)
    elif client is None:
        st.info("Add ANTHROPIC_API_KEY to enable the AI training plan generator.")
