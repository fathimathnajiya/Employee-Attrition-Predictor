import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")

# ---- look and feel (works in light and dark mode) ----
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #1f3c88 0%, #4f86f7 100%); color: white;
       padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {color: #e6eeff; margin: .3rem 0 0 0;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 16px; padding: 1.1rem 1.3rem;
       background: rgba(128,128,128,.08); height: 100%;}
.card .label {font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em;}
.card .big {font-size: 2.8rem; font-weight: 700; line-height: 1.15;}
.pill {display: inline-block; padding: .15rem .8rem; border-radius: 999px; color: white;
       font-weight: 600; font-size: .9rem;}
.bar {height: 12px; border-radius: 8px; background: rgba(128,128,128,.25); overflow: hidden; margin: .6rem 0;}
.bar > div {height: 100%; border-radius: 8px;}
.seg {border-left: 6px solid; border-radius: 14px; padding: .9rem 1.1rem;
      background: rgba(128,128,128,.08); border-top: 1px solid rgba(128,128,128,.2);
      border-right: 1px solid rgba(128,128,128,.2); border-bottom: 1px solid rgba(128,128,128,.2);}
.seg {min-height: 11rem;}
.seg .name {font-weight: 700; font-size: 1.05rem;}
.seg .num {font-size: 1.9rem; font-weight: 700;}
.cmt {border-collapse: separate; border-spacing: 8px; width: 100%; text-align: center;}
.cmt th {font-weight: 600; opacity: .8; padding: .3rem;}
.cmt td.cell {border-radius: 14px; padding: 1.1rem .5rem; width: 38%;}
.cmt td.cell b {font-size: 2.2rem; display: block; line-height: 1.2;}
.cmt td.cell small {opacity: .85;}
.cmt td.rowh {font-weight: 600; text-align: right; width: 14%; opacity: .85;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
                          padding: .8rem 1rem; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

BAND_COLOR = {"Low": "#2e9e6b", "Medium": "#e0a100", "High": "#d64545"}
SEG_COLOR = {"Early-career, low pay": "#e07a5f", "Mid-career, new to company": "#3d85c6",
             "Long-serving, stalled promotion": "#8e6bbf", "Senior, high pay": "#2e9e6b"}


# ----------------------------------------------------------------------------
# 1. TRAIN EVERYTHING ONCE (same steps as the notebook). Cached, so the model
#    is trained only the first time the app opens, not on every click.
# ----------------------------------------------------------------------------
@st.cache_resource
def train():
    df = pd.read_csv("hr.csv", encoding="utf-8-sig")

    # clean
    df1 = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
    df1["Attrition"] = df1["Attrition"].map({"No": 0, "Yes": 1})

    # encoding, x and y
    x = pd.get_dummies(df1.drop(columns="Attrition"), drop_first=True, dtype=int)
    y = df1["Attrition"]

    # split, SMOTE (train only), scaling
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y)
    xtrain_sm, ytrain_sm = SMOTE(random_state=42).fit_resample(xtrain, ytrain)
    scaler = StandardScaler()
    xtrain_s = scaler.fit_transform(xtrain_sm)
    xtest_s = scaler.transform(xtest)

    # final model: Logistic Regression
    model = LogisticRegression(max_iter=2000)
    model.fit(xtrain_s, ytrain_sm)
    pred = model.predict(xtest_s)
    proba = model.predict_proba(xtest_s)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(ytest, pred),
        "Recall": recall_score(ytest, pred),
        "Precision": precision_score(ytest, pred),
        "F1": f1_score(ytest, pred),
        "ROC-AUC": roc_auc_score(ytest, proba)
    }
    cm = confusion_matrix(ytest, pred)

    # segments (KMeans)
    seg_cols = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "DistanceFromHome",
                "JobLevel", "JobSatisfaction", "EnvironmentSatisfaction",
                "WorkLifeBalance", "NumCompaniesWorked"]
    seg_df = df1[seg_cols].copy()
    seg_df["OverTime"] = df1["OverTime"].map({"No": 0, "Yes": 1})
    scaler2 = StandardScaler()
    seg_scaled = scaler2.fit_transform(seg_df)
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(seg_scaled)
    df1["Segment"] = kmeans.labels_

    # name each segment from its profile (works whatever the cluster numbers are)
    prof = df1.groupby("Segment")[["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"]].mean()
    names = {}
    names[prof["MonthlyIncome"].idxmax()] = "Senior, high pay"
    names[prof["TotalWorkingYears"].drop(list(names)).idxmin()] = "Early-career, low pay"
    names[prof["YearsAtCompany"].drop(list(names)).idxmax()] = "Long-serving, stalled promotion"
    names[[s for s in prof.index if s not in names][0]] = "Mid-career, new to company"
    df1["SegmentName"] = df1["Segment"].map(names)

    # risk score for every employee
    df1["RiskScore"] = model.predict_proba(scaler.transform(x))[:, 1]

    # default values for fields the user does not fill in
    defaults = {}
    for c in df1.drop(columns=["Attrition", "Segment", "SegmentName", "RiskScore"]).columns:
        defaults[c] = df1[c].median() if pd.api.types.is_numeric_dtype(df1[c]) else df1[c].mode()[0]

    return dict(df1=df1, x=x, model=model, scaler=scaler, kmeans=kmeans, scaler2=scaler2,
                seg_cols=seg_cols, names=names, metrics=metrics, cm=cm, defaults=defaults)


A = train()
df1, x = A["df1"], A["x"]

ACTIONS = {
    "Early-career, low pay": "Review pay and overtime; assign a mentor; discuss growth plans.",
    "Mid-career, new to company": "Strengthen onboarding; hold regular check-ins in the first years.",
    "Long-serving, stalled promotion": "Review promotion path and career options; recognise long service.",
    "Senior, high pay": "Low risk. Keep monitoring and keep roles engaging.",
}


def risk_band(p):
    return "High" if p >= 0.6 else ("Medium" if p >= 0.3 else "Low")


# ----------------------------------------------------------------------------
# 2. PAGE
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>👥 Employee Attrition Predictor with Segments</h1>
  <p>Predict who may leave, understand what type of employee they are, and decide what HR should do.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict an employee", "🧩 Segments", "📊 Model performance"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.sidebar.header("Employee details")
    d = A["defaults"]
    inp = {}
    inp["Age"] = st.sidebar.slider("Age", 18, 60, int(d["Age"]))
    inp["MonthlyIncome"] = st.sidebar.slider("Monthly income", 1000, 20000, int(d["MonthlyIncome"]), step=100)
    inp["JobLevel"] = st.sidebar.slider("Job level", 1, 5, int(d["JobLevel"]))
    inp["OverTime"] = st.sidebar.selectbox("Works overtime?", ["No", "Yes"], index=0)
    inp["MaritalStatus"] = st.sidebar.selectbox("Marital status", ["Single", "Married", "Divorced"], index=1)
    inp["BusinessTravel"] = st.sidebar.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
    inp["Department"] = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=1)
    inp["JobSatisfaction"] = st.sidebar.slider("Job satisfaction (1-4)", 1, 4, int(d["JobSatisfaction"]))
    inp["EnvironmentSatisfaction"] = st.sidebar.slider("Environment satisfaction (1-4)", 1, 4, int(d["EnvironmentSatisfaction"]))
    inp["WorkLifeBalance"] = st.sidebar.slider("Work-life balance (1-4)", 1, 4, int(d["WorkLifeBalance"]))
    inp["TotalWorkingYears"] = st.sidebar.slider("Total working years", 0, 40, int(d["TotalWorkingYears"]))
    inp["YearsAtCompany"] = st.sidebar.slider("Years at company", 0, 40, int(d["YearsAtCompany"]))
    inp["YearsSinceLastPromotion"] = st.sidebar.slider("Years since last promotion", 0, 15, int(d["YearsSinceLastPromotion"]))
    inp["JobRole"] = st.sidebar.selectbox("Job role", sorted(df1["JobRole"].unique()),
                                          index=sorted(df1["JobRole"].unique()).index(d["JobRole"]))
    inp["StockOptionLevel"] = st.sidebar.slider("Stock option level (0-3)", 0, 3, int(d["StockOptionLevel"]))
    inp["JobInvolvement"] = st.sidebar.slider("Job involvement (1-4)", 1, 4, int(d["JobInvolvement"]))
    inp["YearsInCurrentRole"] = st.sidebar.slider("Years in current role", 0, 18, int(d["YearsInCurrentRole"]))
    inp["YearsWithCurrManager"] = st.sidebar.slider("Years with current manager", 0, 17, int(d["YearsWithCurrManager"]))
    inp["DistanceFromHome"] = st.sidebar.slider("Distance from home (km)", 1, 29, int(d["DistanceFromHome"]))
    inp["NumCompaniesWorked"] = st.sidebar.slider("Companies worked before", 0, 9, int(d["NumCompaniesWorked"]))

    # build one raw row, then encode it exactly like the training data
    raw = dict(d)
    raw.update(inp)
    raw = pd.DataFrame([raw])
    enc = pd.get_dummies(raw, dtype=int).reindex(columns=x.columns, fill_value=0)

    scaled = A["scaler"].transform(enc)
    prob = A["model"].predict_proba(scaled)[0, 1]
    band = risk_band(prob)

    seg_new = raw[A["seg_cols"]].copy()
    seg_new["OverTime"] = 1 if inp["OverTime"] == "Yes" else 0
    seg_id = int(A["kmeans"].predict(A["scaler2"].transform(seg_new))[0])
    seg_name = A["names"][seg_id]

    col = BAND_COLOR[band]
    msg = {"High": "Recommend a retention conversation soon.",
           "Medium": "Keep an eye on this employee.",
           "Low": "No immediate concern."}[band]
    left_c, right_c = st.columns(2)
    left_c.markdown(f"""
    <div class="card">
      <div class="label">Chance of leaving</div>
      <div class="big" style="color:{col}">{prob * 100:.1f}%</div>
      <div class="bar"><div style="width:{prob * 100:.1f}%; background:{col}"></div></div>
      <span class="pill" style="background:{col}">{band} risk</span>
      <span style="margin-left:.6rem; opacity:.85">{msg}</span>
    </div>""", unsafe_allow_html=True)
    sc_col = SEG_COLOR[seg_name]
    right_c.markdown(f"""
    <div class="card" style="border-left:6px solid {sc_col}">
      <div class="label">Employee segment</div>
      <div class="big" style="font-size:1.7rem; color:{sc_col}">{seg_name}</div>
      <div style="margin-top:.6rem"><b>Suggested HR action:</b><br>{ACTIONS[seg_name]}</div>
    </div>""", unsafe_allow_html=True)

    # simple explanation: coefficient x scaled value = push towards leaving or staying
    st.markdown("#### Why this prediction?")
    contrib = pd.Series(scaled[0] * A["model"].coef_[0], index=x.columns)
    shown = [c for c in x.columns if any(c == k or c.startswith(k + "_") for k in inp)]
    contrib = contrib[shown].sort_values()
    left, right = st.columns(2)
    left.write("**Factors pushing risk UP**")
    up = contrib.tail(5)[::-1].round(2).rename("push").reset_index()
    up.columns = ["Factor", "push"]
    left.bar_chart(up, x="Factor", y="push", color="#d64545", horizontal=True, sort="-push")
    right.write("**Factors pushing risk DOWN**")
    down = contrib.head(5).round(2).rename("push").reset_index()
    down.columns = ["Factor", "push"]
    right.bar_chart(down, x="Factor", y="push", color="#2e9e6b", horizontal=True, sort="push")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Employee segments")
    order = ["Early-career, low pay", "Mid-career, new to company",
             "Long-serving, stalled promotion", "Senior, high pay"]
    prof = df1.groupby("SegmentName").agg(
        employees=("Attrition", "size"),
        attrition_pct=("Attrition", lambda s: round(s.mean() * 100, 1)),
        avg_risk=("RiskScore", lambda s: round(s.mean(), 2)),
        age=("Age", "mean"), income=("MonthlyIncome", "mean"),
        working_years=("TotalWorkingYears", "mean"), years_at_company=("YearsAtCompany", "mean"),
        years_since_promotion=("YearsSinceLastPromotion", "mean"),
    ).round(1).reindex(order)
    cols4 = st.columns(4)
    for cc, nm in zip(cols4, order):
        r = prof.loc[nm]
        cc.markdown(f"""
        <div class="seg" style="border-left-color:{SEG_COLOR[nm]}">
          <div class="name">{nm}</div>
          <div class="num" style="color:{SEG_COLOR[nm]}">{r['attrition_pct']}%</div>
          <div style="opacity:.8">left the company<br>{int(r['employees'])} employees</div>
        </div>""", unsafe_allow_html=True)
    st.write("")
    st.dataframe(prof)
    chart = prof["attrition_pct"].reset_index()
    chart.columns = ["Segment", "Attrition %"]
    st.bar_chart(chart, x="Segment", y="Attrition %", color="#4f86f7", horizontal=True, sort="-Attrition %")
    st.caption("Bars show the % of employees in each segment who actually left.")

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Model performance (test set, 441 employees)")
    mm = A["metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{mm['Accuracy']:.2f}")
    k2.metric("Recall", f"{mm['Recall']:.2f}")
    k3.metric("Precision", f"{mm['Precision']:.2f}")
    k4.metric("F1", f"{mm['F1']:.2f}")
    k5.metric("ROC-AUC", f"{mm['ROC-AUC']:.2f}")
    cm = A["cm"]
    st.write(f"Caught **{cm[1, 1]} of {cm[1].sum()}** real leavers, with **{cm[0, 1]}** false alarms.")
    st.markdown("#### Confusion matrix")
    rows = [("Actually stayed", 0, ["Correct: stayed", "False alarm"]),
            ("Actually left", 1, ["Missed leaver", "Correct: caught"])]
    html = '<table class="cmt"><tr><th></th><th>Predicted: stay</th><th>Predicted: leave</th></tr>'
    for label, r, notes in rows:
        html += f'<tr><td class="rowh">{label}<br><small>({cm[r].sum()} employees)</small></td>'
        for c in (0, 1):
            share = cm[r, c] / cm[r].sum()
            good = (r == c)
            rgb = "46,158,107" if good else "214,69,69"
            alpha = 0.15 + 0.75 * share if good else 0.15 + 0.75 * min(share * 2, 1)
            html += (f'<td class="cell" style="background:rgba({rgb},{alpha:.2f})">'
                     f'<b>{cm[r, c]}</b><small>{share:.0%} of row<br>{notes[c]}</small></td>')
        html += '</tr>'
    st.markdown(html + '</table>', unsafe_allow_html=True)
    st.caption("Each row adds up to 100%. Green cells are correct predictions, red cells are mistakes. "
               "The darker the cell, the larger its share of that row.")
