# ============================================================
# BANGLADESH NATIONAL ELECTION 2026 — FULL ANALYSIS
# Google Colab Notebook
# ============================================================
# INSTRUCTIONS: Upload election.csv and gonovote.csv before running.
# Run each cell sequentially. All charts auto-save to your session.
# ============================================================

# ─────────────────────────────────────────────
# CELL 1 ─ Install & Import Libraries
# ─────────────────────────────────────────────
!pip install pyspark -q
!pip install folium branca -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
)
from pyspark.ml import Pipeline

import folium
from folium.plugins import MarkerCluster
import json

# ─── Plot Style ───────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
})
PALETTE = {
    'BNP':    '#1f77b4',
    'Jamaat': '#2ca02c',
    'NCP':    '#ff7f0e',
    'IND':    '#9467bd',
    'Other':  '#8c564b',
}
BD_DIVISIONS = {
    'Rangpur':    (25.745,  89.275),
    'Rajshahi':   (24.367,  88.600),
    'Khulna':     (22.845,  89.568),
    'Barishal':   (22.700,  90.370),
    'Mymensingh': (24.750,  90.407),
    'Dhaka':      (23.810,  90.407),
    'Sylhet':     (24.900,  91.872),
    'Chittagong': (22.356,  91.784),
}

print("✅ Libraries loaded successfully.")


# ─────────────────────────────────────────────
# CELL 2 ─ Upload Data Files
# ─────────────────────────────────────────────
from google.colab import files

print("📂 Please upload election.csv ...")
uploaded = files.upload()
print("📂 Please upload gonovote.csv ...")
uploaded2 = files.upload()

print("✅ Files uploaded.")


# ─────────────────────────────────────────────
# CELL 3 ─ Load & Clean Election Data
# ─────────────────────────────────────────────
raw = pd.read_csv('election.csv', header=None)

# Row 1 is the true header
raw.columns = raw.iloc[1]
raw = raw.iloc[2:].reset_index(drop=True)

raw.columns = [
    'Constituency_No', 'Division', 'Constituency_Name',
    'Poverty_Rate', 'Literacy_Rate',
    'Winner_Candidate', 'Winner_Party', 'Winner_Votes',
    'Runner_Candidate', 'Runner_Party', 'Runner_Votes',
    'Margin', 'Total_Voters', 'Male_Voters',
    'Female_Voters', 'Transgender_Voters'
]

# Drop election-postponed rows
df = raw[raw['Winner_Party'].notna() &
         (raw['Winner_Candidate'] != 'Election postponed')].copy()

# Numeric conversion helper
def to_num(col):
    return pd.to_numeric(
        col.astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )

for c in ['Constituency_No','Poverty_Rate','Literacy_Rate',
          'Winner_Votes','Runner_Votes','Margin',
          'Total_Voters','Male_Voters','Female_Voters','Transgender_Voters']:
    df[c] = to_num(df[c])

df['Division'] = df['Division'].str.strip()

# ── Derived Features ──────────────────────────
df['Total_Cast_Votes'] = df['Winner_Votes'] + df['Runner_Votes']
df['Voter_Turnout_Pct'] = (df['Total_Cast_Votes'] / df['Total_Voters'] * 100).round(2)
df['Male_Pct']          = (df['Male_Voters']        / df['Total_Voters'] * 100).round(2)
df['Female_Pct']        = (df['Female_Voters']      / df['Total_Voters'] * 100).round(2)
df['Trans_Pct']         = (df['Transgender_Voters'] / df['Total_Voters'] * 100).round(4)
df['Winning_Margin_Pct']= (df['Margin']             / df['Winner_Votes'] * 100).round(2)

# Simplify party labels
def simplify_party(p):
    p = str(p).strip()
    if p in PALETTE:
        return p
    return 'Other'

df['Winner_Party_Simple'] = df['Winner_Party'].apply(simplify_party)

# Add division coordinates
df['Div_Lat'] = df['Division'].map(lambda d: BD_DIVISIONS.get(d, (0,0))[0])
df['Div_Lon'] = df['Division'].map(lambda d: BD_DIVISIONS.get(d, (0,0))[1])

print(f"✅ Cleaned dataset: {df.shape[0]} constituencies, {df.shape[1]} features")
print(df[['Division','Constituency_Name','Winner_Party',
          'Voter_Turnout_Pct','Male_Pct','Female_Pct','Trans_Pct']].head())


# ─────────────────────────────────────────────
# CELL 4 ─ Load & Display Gonovote Data
# ─────────────────────────────────────────────
gv = pd.read_csv('gonovote.csv')
gv.columns = ['Choice', 'Votes', 'Percentage']
gv['Votes'] = to_num(gv['Votes'])
gv['Percentage'] = gv['Percentage'].astype(str).str.replace('%','').str.strip().astype(float)

print("📊 Gonovote (গণভোট) Summary:")
print(gv.to_string(index=False))


# ─────────────────────────────────────────────
# CELL 5 ─ VOTER ANALYTICS (Male / Female / Trans)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Bangladesh Election 2026 — Voter Demographics", fontsize=16, fontweight='bold')

# --- 5A. National aggregate pie ---------------
total_male  = df['Male_Voters'].sum()
total_fem   = df['Female_Voters'].sum()
total_trans = df['Transgender_Voters'].sum()
sizes       = [total_male, total_fem, total_trans]
labels      = [
    f"Male\n{total_male/1e6:.2f}M\n({total_male/(total_male+total_fem+total_trans)*100:.1f}%)",
    f"Female\n{total_fem/1e6:.2f}M\n({total_fem/(total_male+total_fem+total_trans)*100:.1f}%)",
    f"Transgender\n{total_trans:,}\n({total_trans/(total_male+total_fem+total_trans)*100:.4f}%)"
]
colors = ['#4472C4', '#ED7D31', '#A9D18E']
axes[0].pie(sizes, labels=labels, colors=colors, startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':1.5})
axes[0].set_title("National Voter Distribution", fontsize=13)

# --- 5B. Division-level grouped bar -----------
div_g = df.groupby('Division')[['Male_Voters','Female_Voters','Transgender_Voters']].sum()
div_g = div_g / 1e6   # millions
div_g.plot(kind='bar', ax=axes[1],
           color=['#4472C4','#ED7D31','#A9D18E'], edgecolor='white')
axes[1].set_title("Voters by Division (Millions)", fontsize=13)
axes[1].set_xlabel("")
axes[1].set_ylabel("Voters (Millions)")
axes[1].legend(['Male','Female','Transgender'], loc='upper right')
axes[1].tick_params(axis='x', rotation=45)

# --- 5C. Male vs Female scatter by constituency
sc = axes[2].scatter(df['Male_Pct'], df['Female_Pct'],
                     c=df['Transgender_Voters'], cmap='YlOrRd',
                     alpha=0.7, edgecolors='grey', linewidth=0.3, s=40)
plt.colorbar(sc, ax=axes[2], label='Transgender Voters')
axes[2].set_xlabel("Male Voters %")
axes[2].set_ylabel("Female Voters %")
axes[2].set_title("Male vs Female % per Constituency", fontsize=13)
axes[2].axline((50, 50), slope=1, color='red', linestyle='--', linewidth=0.8, label='Equal line')
axes[2].legend()

plt.tight_layout()
plt.savefig('voter_demographics.png', bbox_inches='tight')
plt.show()

# Summary stats
print("\n📊 Voter Gender Summary:")
print(f"  Total Male Voters      : {total_male:>12,}  ({total_male/(total_male+total_fem+total_trans)*100:.2f}%)")
print(f"  Total Female Voters    : {total_fem:>12,}  ({total_fem/(total_male+total_fem+total_trans)*100:.2f}%)")
print(f"  Total Transgender      : {total_trans:>12,}  ({total_trans/(total_male+total_fem+total_trans)*100:.4f}%)")


# ─────────────────────────────────────────────
# CELL 6 ─ ELECTION OVERVIEW — Party Results
# ─────────────────────────────────────────────
party_counts = df['Winner_Party'].value_counts().reset_index()
party_counts.columns = ['Party', 'Seats']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Bangladesh National Election 2026 — Party Seat Distribution",
             fontsize=15, fontweight='bold')

# Pie chart (top 8 + Others)
top_n = 7
top    = party_counts.head(top_n)
others = pd.DataFrame({'Party': ['Others'], 'Seats': [party_counts.iloc[top_n:]['Seats'].sum()]})
pie_df = pd.concat([top, others], ignore_index=True)

cmap = plt.cm.get_cmap('tab10', len(pie_df))
pie_colors = [cmap(i) for i in range(len(pie_df))]
axes[0].pie(pie_df['Seats'], labels=pie_df['Party'], colors=pie_colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor':'white','linewidth':1.2})
axes[0].set_title("Seat Share by Party")

# Bar chart
colors_bar = [PALETTE.get(p, '#8c564b') for p in party_counts.head(12)['Party']]
axes[1].barh(party_counts.head(12)['Party'][::-1],
             party_counts.head(12)['Seats'][::-1],
             color=colors_bar[::-1], edgecolor='white')
axes[1].set_xlabel("Number of Seats Won")
axes[1].set_title("Top 12 Parties — Seats Won")
for i, (v, p) in enumerate(zip(party_counts.head(12)['Seats'][::-1],
                                party_counts.head(12)['Party'][::-1])):
    axes[1].text(v + 0.5, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig('party_seat_distribution.png', bbox_inches='tight')
plt.show()

print("\n📊 Top 10 Parties:")
print(party_counts.head(10).to_string(index=False))


# ─────────────────────────────────────────────
# CELL 7 ─ GONOVOTE (গণভোট) ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("গণভোট (Gonovote / People's Vote) 2026 — Analysis", fontsize=15, fontweight='bold')

# Pie: Yes vs No
yes_no = gv[gv['Choice'].isin(['Yes','No'])]
axes[0].pie(yes_no['Votes'], labels=yes_no['Choice'],
            autopct='%1.2f%%', colors=['#2ca02c','#d62728'],
            startangle=90, wedgeprops={'edgecolor':'white','linewidth':2},
            textprops={'fontsize':13})
axes[0].set_title("Yes vs No (Valid Votes Only)", fontsize=13)

# Bar: Full breakdown
full = gv.iloc[:5].copy()
bar_colors = ['#2ca02c','#d62728','#aec7e8','#ffbb78','#1f77b4']
bars = axes[1].bar(full['Choice'], full['Votes']/1e6, color=bar_colors, edgecolor='white', linewidth=1.2)
axes[1].set_ylabel("Votes (Millions)")
axes[1].set_title("Gonovote Breakdown (Millions)", fontsize=13)
axes[1].tick_params(axis='x', rotation=15)
for bar, row in zip(bars, full.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{row.Votes/1e6:.2f}M\n({row.Percentage}%)",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('gonovote_analysis.png', bbox_inches='tight')
plt.show()

print("\n📋 Gonovote Key Statistics:")
total_votes = gv[gv['Choice']=='Total votes']['Votes'].values[0]
valid_votes  = gv[gv['Choice']=='Valid votes']['Votes'].values[0]
yes_votes    = gv[gv['Choice']=='Yes']['Votes'].values[0]
no_votes     = gv[gv['Choice']=='No']['Votes'].values[0]
invalid_v    = gv[gv['Choice']=='Invalid or blank votes']['Votes'].values[0]
print(f"  Total Votes Cast   : {total_votes:>12,}")
print(f"  Valid Votes        : {valid_votes:>12,}  ({valid_votes/total_votes*100:.2f}%)")
print(f"  Invalid / Blank    : {invalid_v:>12,}  ({invalid_v/total_votes*100:.2f}%)")
print(f"  YES Votes          : {yes_votes:>12,}  ({yes_votes/valid_votes*100:.2f}% of valid)")
print(f"  NO  Votes          : {no_votes:>12,}  ({no_votes/valid_votes*100:.2f}% of valid)")


# ─────────────────────────────────────────────
# CELL 8 ─ GEOGRAPHIC VISUALIZATION (Folium Map)
# ─────────────────────────────────────────────
# Division-level summary
div_summary = df.groupby('Division').agg(
    Total_Seats=('Constituency_No','count'),
    BNP_Seats=('Winner_Party', lambda x: (x=='BNP').sum()),
    Jamaat_Seats=('Winner_Party', lambda x: (x=='Jamaat').sum()),
    Avg_Turnout=('Voter_Turnout_Pct','mean'),
    Avg_Literacy=('Literacy_Rate','mean'),
    Avg_Poverty=('Poverty_Rate','mean'),
).reset_index()

bd_map = folium.Map(location=[23.68, 90.35], zoom_start=7, tiles='CartoDB positron')

# Color mapping for leading party
def lead_color(row):
    if row['BNP_Seats'] >= row['Jamaat_Seats']:
        return '#1f77b4'   # BNP blue
    return '#2ca02c'       # Jamaat green

for _, row in div_summary.iterrows():
    lat, lon = BD_DIVISIONS.get(row['Division'], (23.68, 90.35))
    color = lead_color(row)
    popup_html = f"""
    <div style='font-family:Arial;min-width:200px'>
      <b style='font-size:14px'>{row['Division']} Division</b><br>
      <hr style='margin:4px 0'>
      Total Seats: <b>{int(row['Total_Seats'])}</b><br>
      BNP Seats: <b style='color:#1f77b4'>{int(row['BNP_Seats'])}</b><br>
      Jamaat Seats: <b style='color:#2ca02c'>{int(row['Jamaat_Seats'])}</b><br>
      Avg Turnout: <b>{row['Avg_Turnout']:.1f}%</b><br>
      Avg Literacy: <b>{row['Avg_Literacy']:.1f}%</b><br>
      Avg Poverty: <b>{row['Avg_Poverty']:.1f}%</b>
    </div>
    """
    folium.CircleMarker(
        location=[lat, lon],
        radius=max(row['Total_Seats']/2, 10),
        color=color, fill=True, fill_color=color, fill_opacity=0.6,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['Division']}: {int(row['BNP_Seats'])} BNP | {int(row['Jamaat_Seats'])} Jamaat"
    ).add_to(bd_map)

# Constituency-level markers
mc = MarkerCluster().add_to(bd_map)
for _, row in df.iterrows():
    lat = row['Div_Lat'] + np.random.uniform(-0.3, 0.3)
    lon = row['Div_Lon'] + np.random.uniform(-0.3, 0.3)
    color = PALETTE.get(row['Winner_Party'], 'gray')
    folium.CircleMarker(
        location=[lat, lon], radius=4,
        color=color, fill=True, fill_color=color, fill_opacity=0.8,
        tooltip=(f"{row['Constituency_Name']} | {row['Winner_Party']} | "
                 f"Turnout: {row['Voter_Turnout_Pct']:.1f}%"),
        popup=folium.Popup(
            f"<b>{row['Constituency_Name']}</b><br>"
            f"Winner: {row['Winner_Candidate']} ({row['Winner_Party']})<br>"
            f"Votes: {int(row['Winner_Votes']):,} | Margin: {int(row['Margin']):,}<br>"
            f"Turnout: {row['Voter_Turnout_Pct']:.1f}% | "
            f"Literacy: {row['Literacy_Rate']}% | Poverty: {row['Poverty_Rate']}%",
            max_width=300)
    ).add_to(mc)

# Legend
legend_html = """
<div style='position:fixed;bottom:30px;left:30px;background:white;
            padding:10px;border-radius:8px;border:1px solid #ccc;
            font-family:Arial;font-size:12px;z-index:9999'>
  <b>Winner Party</b><br>
  <span style='color:#1f77b4'>●</span> BNP<br>
  <span style='color:#2ca02c'>●</span> Jamaat<br>
  <span style='color:#ff7f0e'>●</span> NCP<br>
  <span style='color:#9467bd'>●</span> Independent<br>
  <span style='color:#8c564b'>●</span> Other
</div>
"""
bd_map.get_root().html.add_child(folium.Element(legend_html))
bd_map.save('bangladesh_election_map.html')
print("✅ Interactive map saved → bangladesh_election_map.html")
bd_map


# ─────────────────────────────────────────────
# CELL 9 ─ CHARTS & GRAPHS — Voter Turnout
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Voter Turnout & Electoral Analysis — Bangladesh 2026",
             fontsize=15, fontweight='bold')

# 9A. Turnout by Division (box plot)
div_order = df.groupby('Division')['Voter_Turnout_Pct'].median().sort_values(ascending=False).index
df_plot = df[df['Division'].isin(div_order)]
bp_data = [df_plot[df_plot['Division']==d]['Voter_Turnout_Pct'].dropna().values
           for d in div_order]
bp = axes[0,0].boxplot(bp_data, labels=div_order, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
colors_div = plt.cm.Set2(np.linspace(0, 1, len(div_order)))
for patch, color in zip(bp['boxes'], colors_div):
    patch.set_facecolor(color)
axes[0,0].set_title("Voter Turnout by Division (%)")
axes[0,0].set_ylabel("Turnout %")
axes[0,0].tick_params(axis='x', rotation=45)

# 9B. Turnout by Winning Party
party_turnout = df.groupby('Winner_Party')['Voter_Turnout_Pct'].mean().sort_values(ascending=False).head(10)
colors_pt = [PALETTE.get(p, '#8c564b') for p in party_turnout.index]
axes[0,1].bar(party_turnout.index, party_turnout.values, color=colors_pt, edgecolor='white')
axes[0,1].set_title("Avg Voter Turnout by Winning Party")
axes[0,1].set_ylabel("Avg Turnout %")
axes[0,1].tick_params(axis='x', rotation=45)
for i, v in enumerate(party_turnout.values):
    axes[0,1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)

# 9C. Winning Margin Distribution
axes[1,0].hist(df['Margin'].dropna(), bins=40, color='#4472C4', edgecolor='white', alpha=0.85)
axes[1,0].axvline(df['Margin'].median(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Median: {df['Margin'].median():,.0f}")
axes[1,0].set_title("Distribution of Winning Margins")
axes[1,0].set_xlabel("Winning Margin (Votes)")
axes[1,0].set_ylabel("Frequency")
axes[1,0].legend()

# 9D. Top 15 — Highest & Lowest Margins
top15_high = df.nlargest(8, 'Margin')[['Constituency_Name','Winner_Party','Margin']]
top15_low  = df.nsmallest(7, 'Margin')[['Constituency_Name','Winner_Party','Margin']]
combined   = pd.concat([top15_high, top15_low]).reset_index(drop=True)
bar_colors_c = ['#2ca02c' if v > 50000 else '#d62728' for v in combined['Margin']]
axes[1,1].barh(combined['Constituency_Name'], combined['Margin'],
               color=bar_colors_c, edgecolor='white')
axes[1,1].set_title("Top Margins (Green=High, Red=Tight)")
axes[1,1].set_xlabel("Margin (Votes)")
axes[1,1].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('voter_turnout_charts.png', bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# CELL 10 ─ CORRELATION ANALYSIS (Pearson & Spearman)
# ─────────────────────────────────────────────
# 10A. Compute using both pandas and PySpark

# --- Pandas correlations ----------------------
corr_features = ['Poverty_Rate','Literacy_Rate','Voter_Turnout_Pct',
                  'Male_Pct','Female_Pct','Margin','Total_Voters']
corr_df = df[corr_features].dropna()

pearson_matrix  = corr_df.corr(method='pearson')
spearman_matrix = corr_df.corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Correlation Analysis — Socio-Economic vs Electoral Factors",
             fontsize=15, fontweight='bold')

sns.heatmap(pearson_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[0], square=True, linewidths=0.5,
            cbar_kws={'shrink':0.8})
axes[0].set_title("Pearson Correlation Matrix")

sns.heatmap(spearman_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[1], square=True, linewidths=0.5,
            cbar_kws={'shrink':0.8})
axes[1].set_title("Spearman Correlation Matrix")

plt.tight_layout()
plt.savefig('correlation_heatmaps.png', bbox_inches='tight')
plt.show()

# --- PySpark Correlation (MLlib) ---------------
spark = SparkSession.builder.appName("BD_Election_2026").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sdf = spark.createDataFrame(corr_df.fillna(0))
va  = VectorAssembler(inputCols=corr_features, outputCol='features')
sdf_vec = va.transform(sdf).select('features')

pcc_result = Correlation.corr(sdf_vec, 'features', 'pearson')
scc_result = Correlation.corr(sdf_vec, 'features', 'spearman')

pcc_arr = np.array(pcc_result.collect()[0][0].toArray())
scc_arr = np.array(scc_result.collect()[0][0].toArray())

print("\n📊 PySpark (MLlib) Pearson Correlation — Literacy Rate vs Voter Turnout:")
lit_idx = corr_features.index('Literacy_Rate')
trn_idx = corr_features.index('Voter_Turnout_Pct')
pov_idx = corr_features.index('Poverty_Rate')

print(f"  Literacy  ↔ Voter Turnout  (Pearson)  : r = {pcc_arr[lit_idx, trn_idx]:.4f}")
print(f"  Poverty   ↔ Voter Turnout  (Pearson)  : r = {pcc_arr[pov_idx, trn_idx]:.4f}")
print(f"  Literacy  ↔ Voter Turnout  (Spearman) : r = {scc_arr[lit_idx, trn_idx]:.4f}")
print(f"  Poverty   ↔ Voter Turnout  (Spearman) : r = {scc_arr[pov_idx, trn_idx]:.4f}")


# ─────────────────────────────────────────────
# CELL 11 ─ GEOGRAPHIC CORRELATION — Division & Winning Party
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Geographic Position vs Winning Party", fontsize=14, fontweight='bold')

# 11A. Party wins by division (stacked bar)
div_party = df.groupby(['Division','Winner_Party_Simple']).size().unstack(fill_value=0)
party_cols = [p for p in ['BNP','Jamaat','NCP','IND','Other'] if p in div_party.columns]
div_party[party_cols].plot(kind='bar', stacked=True, ax=axes[0],
                            color=[PALETTE[p] for p in party_cols], edgecolor='white')
axes[0].set_title("Seats Won by Division & Party")
axes[0].set_xlabel("")
axes[0].set_ylabel("Number of Seats")
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(loc='upper right', title='Party')

# 11B. Latitude vs BNP win rate scatter
div_stats = df.groupby('Division').agg(
    BNP_Rate=('Winner_Party', lambda x: (x=='BNP').mean() * 100),
    Jamaat_Rate=('Winner_Party', lambda x: (x=='Jamaat').mean() * 100),
    Avg_Lat=('Div_Lat', 'mean'),
    Avg_Lon=('Div_Lon', 'mean'),
    Avg_Poverty=('Poverty_Rate', 'mean'),
).reset_index()

sc1 = axes[1].scatter(div_stats['Avg_Lat'], div_stats['BNP_Rate'],
                       c=div_stats['Avg_Poverty'], cmap='Reds',
                       s=200, edgecolors='grey', linewidth=0.5, zorder=3)
axes[1].scatter(div_stats['Avg_Lat'], div_stats['Jamaat_Rate'],
                c=div_stats['Avg_Poverty'], cmap='Greens',
                s=200, marker='D', edgecolors='grey', linewidth=0.5, zorder=3)
for _, row in div_stats.iterrows():
    axes[1].annotate(row['Division'], (row['Avg_Lat'], row['BNP_Rate']),
                     fontsize=7, ha='left', va='bottom')
plt.colorbar(sc1, ax=axes[1], label='Avg Poverty Rate')
axes[1].set_xlabel("Latitude (North ↑)")
axes[1].set_ylabel("Party Win Rate (%)")
axes[1].set_title("Latitude vs Win Rate (● BNP | ◆ Jamaat)")

plt.tight_layout()
plt.savefig('geographic_correlation.png', bbox_inches='tight')
plt.show()

# Numerical correlation
print("\n📊 Correlation — Geographic Position vs Winning Party:")
df['BNP_Win'] = (df['Winner_Party'] == 'BNP').astype(int)
print(f"  Latitude  ↔ BNP Win   (Pearson) : r = {df['Div_Lat'].corr(df['BNP_Win']):.4f}")
print(f"  Longitude ↔ BNP Win   (Pearson) : r = {df['Div_Lon'].corr(df['BNP_Win']):.4f}")


# ─────────────────────────────────────────────
# CELL 12 ─ LITERACY RATE vs WINNING PARTY
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Literacy Rate vs Winning Party", fontsize=14, fontweight='bold')

# 12A. Literacy rate distribution per party (violin)
top_parties = df['Winner_Party_Simple'].value_counts().head(4).index
df_top = df[df['Winner_Party_Simple'].isin(top_parties)]
party_literacy = [df_top[df_top['Winner_Party_Simple']==p]['Literacy_Rate'].dropna().values
                  for p in top_parties]
vp = axes[0].violinplot(party_literacy, positions=range(len(top_parties)),
                         showmedians=True, showmeans=False)
for i, (pc, p) in enumerate(zip(vp['bodies'], top_parties)):
    pc.set_facecolor(PALETTE.get(p, '#8c564b'))
    pc.set_alpha(0.7)
axes[0].set_xticks(range(len(top_parties)))
axes[0].set_xticklabels(top_parties)
axes[0].set_ylabel("Literacy Rate (%)")
axes[0].set_title("Literacy Rate Distribution by Winning Party")

# 12B. Avg literacy rate per party bar
lit_by_party = df.groupby('Winner_Party_Simple')['Literacy_Rate'].mean().sort_values(ascending=False)
bar_c = [PALETTE.get(p, '#8c564b') for p in lit_by_party.index]
axes[1].bar(lit_by_party.index, lit_by_party.values, color=bar_c, edgecolor='white')
axes[1].set_ylabel("Avg Literacy Rate (%)")
axes[1].set_title("Average Literacy Rate by Winning Party")
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(lit_by_party.values):
    axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('literacy_vs_party.png', bbox_inches='tight')
plt.show()

# Point-biserial correlation
from scipy.stats import pointbiserialr
df['Jamaat_Win'] = (df['Winner_Party'] == 'Jamaat').astype(int)
bnp_r,   bnp_p   = pointbiserialr(df['Literacy_Rate'].fillna(df['Literacy_Rate'].mean()), df['BNP_Win'])
jam_r,   jam_p   = pointbiserialr(df['Literacy_Rate'].fillna(df['Literacy_Rate'].mean()), df['Jamaat_Win'])
print("\n📊 Point-Biserial Correlation — Literacy Rate & Party Win:")
print(f"  Literacy ↔ BNP Win    : r = {bnp_r:.4f}  (p = {bnp_p:.4f})")
print(f"  Literacy ↔ Jamaat Win : r = {jam_r:.4f}  (p = {jam_p:.4f})")


# ─────────────────────────────────────────────
# CELL 13 ─ POVERTY vs WINNING PARTY
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Poverty Rate vs Winning Party", fontsize=14, fontweight='bold')

# 13A. Scatter — poverty vs turnout, colored by party
for party in ['BNP','Jamaat','NCP','IND','Other']:
    sub = df[df['Winner_Party_Simple'] == party]
    axes[0].scatter(sub['Poverty_Rate'], sub['Voter_Turnout_Pct'],
                    label=party, color=PALETTE[party], alpha=0.6, s=30)
# Trend line
valid = df[['Poverty_Rate','Voter_Turnout_Pct']].dropna()
m, b = np.polyfit(valid['Poverty_Rate'], valid['Voter_Turnout_Pct'], 1)
xs = np.linspace(valid['Poverty_Rate'].min(), valid['Poverty_Rate'].max(), 100)
axes[0].plot(xs, m*xs + b, 'k--', linewidth=1.5, label=f'Trend (r={valid.corr().iloc[0,1]:.2f})')
axes[0].set_xlabel("Poverty Rate (HCR %)")
axes[0].set_ylabel("Voter Turnout %")
axes[0].set_title("Poverty Rate vs Voter Turnout (by Party)")
axes[0].legend(loc='upper right', fontsize=8)

# 13B. Avg poverty by winning party
pov_party = df.groupby('Winner_Party_Simple')['Poverty_Rate'].mean().sort_values(ascending=False)
bar_c2 = [PALETTE.get(p,'#8c564b') for p in pov_party.index]
axes[1].bar(pov_party.index, pov_party.values, color=bar_c2, edgecolor='white')
axes[1].set_ylabel("Avg Poverty Rate (HCR %)")
axes[1].set_title("Average Poverty Rate by Winning Party")
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(pov_party.values):
    axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('poverty_vs_party.png', bbox_inches='tight')
plt.show()

pov_bnp_r, pov_bnp_p   = pointbiserialr(df['Poverty_Rate'].fillna(df['Poverty_Rate'].mean()), df['BNP_Win'])
pov_jam_r, pov_jam_p   = pointbiserialr(df['Poverty_Rate'].fillna(df['Poverty_Rate'].mean()), df['Jamaat_Win'])
print("\n📊 Point-Biserial Correlation — Poverty Rate & Party Win:")
print(f"  Poverty ↔ BNP Win    : r = {pov_bnp_r:.4f}  (p = {pov_bnp_p:.4f})")
print(f"  Poverty ↔ Jamaat Win : r = {pov_jam_r:.4f}  (p = {pov_jam_p:.4f})")


# ─────────────────────────────────────────────
# CELL 14 ─ REGRESSION ANALYSIS (PySpark MLlib)
# Predict Voter Turnout from Socio-Economic Features
# ─────────────────────────────────────────────
from pyspark.sql.functions import col, when

reg_features = ['Poverty_Rate','Literacy_Rate','Total_Voters','Margin']
reg_df_pd = df[reg_features + ['Voter_Turnout_Pct']].dropna()

reg_sdf = spark.createDataFrame(reg_df_pd)

# Cast to double
for c in reg_features + ['Voter_Turnout_Pct']:
    reg_sdf = reg_sdf.withColumn(c, col(c).cast(DoubleType()))

va_reg = VectorAssembler(inputCols=reg_features, outputCol='raw_features')
scaler = StandardScaler(inputCol='raw_features', outputCol='features', withStd=True, withMean=True)
lr     = LinearRegression(featuresCol='features', labelCol='Voter_Turnout_Pct',
                           maxIter=100, regParam=0.01)

pipe_reg = Pipeline(stages=[va_reg, scaler, lr])

train_sdf, test_sdf = reg_sdf.randomSplit([0.8, 0.2], seed=42)
reg_model = pipe_reg.fit(train_sdf)
preds_reg  = reg_model.transform(test_sdf)

eval_rmse = RegressionEvaluator(labelCol='Voter_Turnout_Pct', metricName='rmse')
eval_r2   = RegressionEvaluator(labelCol='Voter_Turnout_Pct', metricName='r2')
eval_mae  = RegressionEvaluator(labelCol='Voter_Turnout_Pct', metricName='mae')

rmse = eval_rmse.evaluate(preds_reg)
r2   = eval_r2.evaluate(preds_reg)
mae  = eval_mae.evaluate(preds_reg)

lr_model = reg_model.stages[-1]
coefs    = dict(zip(reg_features, lr_model.coefficients.toArray()))

print("\n📊 Linear Regression — Predict Voter Turnout")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f} percentage points")
print(f"  MAE  = {mae:.4f} percentage points")
print("\n  Coefficients (standardized):")
for feat, val in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if val > 0 else "↓"
    print(f"    {feat:<25} : {val:+.4f}  {direction}")
print(f"\n  Intercept: {lr_model.intercept:.4f}")

# Actual vs Predicted Plot
preds_pd = preds_reg.select('Voter_Turnout_Pct','prediction').toPandas()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Regression Analysis — Voter Turnout Prediction", fontsize=14, fontweight='bold')

axes[0].scatter(preds_pd['Voter_Turnout_Pct'], preds_pd['prediction'],
                alpha=0.6, color='#4472C4', edgecolors='grey', linewidth=0.3)
min_v, max_v = preds_pd['Voter_Turnout_Pct'].min(), preds_pd['Voter_Turnout_Pct'].max()
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect Fit')
axes[0].set_xlabel("Actual Turnout %")
axes[0].set_ylabel("Predicted Turnout %")
axes[0].set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
axes[0].legend()

# Coefficients
sorted_coefs = sorted(coefs.items(), key=lambda x: x[1])
colors_coef  = ['#d62728' if v < 0 else '#2ca02c' for _, v in sorted_coefs]
axes[1].barh([k for k, v in sorted_coefs], [v for k, v in sorted_coefs],
             color=colors_coef, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=1)
axes[1].set_xlabel("Coefficient Value")
axes[1].set_title("Feature Coefficients (Green=+, Red=−)")

plt.tight_layout()
plt.savefig('regression_analysis.png', bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# CELL 15 ─ CLASSIFICATION — Predict Winning Party
# (Logistic Regression + Random Forest + Decision Tree)
# ─────────────────────────────────────────────
cls_features = ['Poverty_Rate','Literacy_Rate','Voter_Turnout_Pct',
                 'Margin','Total_Voters']

# Binary: BNP=1, Jamaat=0  (the two dominant parties)
cls_df = df[df['Winner_Party'].isin(['BNP','Jamaat'])][cls_features + ['Winner_Party']].dropna()
cls_df['label'] = (cls_df['Winner_Party'] == 'BNP').astype(float)

cls_sdf = spark.createDataFrame(cls_df[cls_features + ['label']])
for c in cls_features + ['label']:
    cls_sdf = cls_sdf.withColumn(c, col(c).cast(DoubleType()))

va_cls    = VectorAssembler(inputCols=cls_features, outputCol='raw_features')
scaler_c  = StandardScaler(inputCol='raw_features', outputCol='features', withStd=True, withMean=True)

train_c, test_c = cls_sdf.randomSplit([0.8, 0.2], seed=42)

# Models
lr_cls   = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100)
rf_cls   = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100, seed=42)
dt_cls   = DecisionTreeClassifier(featuresCol='features', labelCol='label', seed=42)

acc_eval = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
f1_eval  = MulticlassClassificationEvaluator(labelCol='label', metricName='f1')
bin_eval = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

results_cls = {}
for name, model_obj in [('Logistic Regression', lr_cls),
                         ('Random Forest',       rf_cls),
                         ('Decision Tree',       dt_cls)]:
    pipe_c = Pipeline(stages=[va_cls, scaler_c, model_obj])
    fitted = pipe_c.fit(train_c)
    preds  = fitted.transform(test_c)
    results_cls[name] = {
        'Accuracy':    acc_eval.evaluate(preds),
        'F1-Score':    f1_eval.evaluate(preds),
        'AUC-ROC':     bin_eval.evaluate(preds),
    }

print("\n📊 Classification Results — Predict BNP vs Jamaat:")
print(f"  {'Model':<22} {'Accuracy':>9} {'F1-Score':>10} {'AUC-ROC':>10}")
print("  " + "-"*55)
for name, metrics in results_cls.items():
    print(f"  {name:<22} {metrics['Accuracy']:>9.4f} {metrics['F1-Score']:>10.4f} {metrics['AUC-ROC']:>10.4f}")

# Feature Importance from Random Forest
rf_pipe = Pipeline(stages=[va_cls, scaler_c, rf_cls])
rf_fit  = rf_pipe.fit(train_c)
importances = rf_fit.stages[-1].featureImportances.toArray()
fi_df = pd.DataFrame({'Feature': cls_features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Classification — Predict Winning Party (BNP vs Jamaat)",
             fontsize=14, fontweight='bold')

# Accuracy comparison
model_names = list(results_cls.keys())
acc_vals    = [results_cls[m]['Accuracy'] for m in model_names]
f1_vals     = [results_cls[m]['F1-Score'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
axes[0].bar(x - w/2, acc_vals, w, label='Accuracy', color='#4472C4', edgecolor='white')
axes[0].bar(x + w/2, f1_vals,  w, label='F1-Score', color='#ED7D31', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=15)
axes[0].set_ylabel("Score")
axes[0].set_title("Model Comparison")
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
    axes[0].text(i - w/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=9)
    axes[0].text(i + w/2, f + 0.01, f'{f:.2f}', ha='center', fontsize=9)

# Feature importance
axes[1].barh(fi_df['Feature'], fi_df['Importance'], color='#2ca02c', edgecolor='white')
axes[1].set_xlabel("Importance")
axes[1].set_title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig('classification_results.png', bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# CELL 16 ─ K-MEANS CLUSTERING
# ─────────────────────────────────────────────
clust_features = ['Poverty_Rate','Literacy_Rate','Voter_Turnout_Pct',
                   'Male_Pct','Female_Pct','Margin']
clust_df = df[clust_features + ['Division','Winner_Party_Simple']].dropna()
clust_sdf = spark.createDataFrame(clust_df[clust_features])
for c in clust_features:
    clust_sdf = clust_sdf.withColumn(c, col(c).cast(DoubleType()))

va_km   = VectorAssembler(inputCols=clust_features, outputCol='raw_features')
sc_km   = StandardScaler(inputCol='raw_features', outputCol='features', withStd=True, withMean=True)

# Elbow method
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km    = KMeans(featuresCol='features', k=k, seed=42)
    pipe  = Pipeline(stages=[va_km, sc_km, km])
    model = pipe.fit(clust_sdf)
    inertias.append(model.stages[-1].summary.trainingCost)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("K-Means Clustering — Constituency Socio-Economic Profiles",
             fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Training Cost)")
axes[0].set_title("Elbow Curve")
axes[0].axvline(3, color='red', linestyle='--', label='Optimal k=3')
axes[0].legend()

# Fit with optimal k=3
km_opt  = KMeans(featuresCol='features', k=3, seed=42)
pipe_km = Pipeline(stages=[va_km, sc_km, km_opt])
km_fit  = pipe_km.fit(clust_sdf)
km_pred = km_fit.transform(clust_sdf)

clust_result = km_pred.select('prediction').toPandas()
clust_df     = clust_df.reset_index(drop=True)
clust_df['Cluster'] = clust_result['prediction']

# Cluster profiles
cluster_profile = clust_df.groupby('Cluster')[clust_features].mean().round(2)
print("\n📊 Cluster Profiles (K=3):")
print(cluster_profile.to_string())

# Cluster scatter — Poverty vs Literacy, colored by cluster
colors_km = ['#e41a1c','#377eb8','#4daf4a']
for c in [0, 1, 2]:
    sub = clust_df[clust_df['Cluster'] == c]
    axes[1].scatter(sub['Poverty_Rate'], sub['Literacy_Rate'],
                    label=f'Cluster {c}', color=colors_km[c], alpha=0.7, s=40)
axes[1].set_xlabel("Poverty Rate (%)")
axes[1].set_ylabel("Literacy Rate (%)")
axes[1].set_title("Clusters: Poverty vs Literacy")
axes[1].legend()

# Cluster vs Party distribution
clu_party = clust_df.groupby(['Cluster','Winner_Party_Simple']).size().unstack(fill_value=0)
clu_cols  = [p for p in ['BNP','Jamaat','NCP','IND','Other'] if p in clu_party.columns]
clu_party[clu_cols].plot(kind='bar', stacked=True, ax=axes[2],
                          color=[PALETTE[p] for p in clu_cols], edgecolor='white')
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("Number of Constituencies")
axes[2].set_title("Cluster vs Winning Party")
axes[2].legend(title='Party', loc='upper right')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('kmeans_clustering.png', bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# CELL 17 ─ COMPREHENSIVE SUMMARY DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle("Bangladesh National Election 2026 — Executive Summary Dashboard",
             fontsize=18, fontweight='bold', y=0.98, color='#1a1a2e')

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.4)

# ── Panel 1: Party Seats (Pie) ────────────────
ax1 = fig.add_subplot(gs[0, 0])
top_p  = party_counts.head(6)
others = pd.DataFrame({'Party':['Others'], 'Seats':[party_counts.iloc[6:]['Seats'].sum()]})
pie_d  = pd.concat([top_p, others], ignore_index=True)
cmap10 = plt.cm.get_cmap('tab10', len(pie_d))
ax1.pie(pie_d['Seats'], labels=pie_d['Party'],
        colors=[cmap10(i) for i in range(len(pie_d))],
        autopct='%1.0f%%', startangle=90, textprops={'fontsize':7},
        wedgeprops={'edgecolor':'white','linewidth':1})
ax1.set_title("Seat Share", fontweight='bold', fontsize=10)

# ── Panel 2: Turnout by Division ─────────────
ax2 = fig.add_subplot(gs[0, 1:3])
div_turnout = df.groupby('Division')['Voter_Turnout_Pct'].mean().sort_values()
colors_d = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(div_turnout)))
ax2.barh(div_turnout.index, div_turnout.values, color=colors_d)
ax2.set_xlabel("Avg Turnout %")
ax2.set_title("Avg Voter Turnout by Division", fontweight='bold', fontsize=10)
for i, v in enumerate(div_turnout.values):
    ax2.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=7)

# ── Panel 3: Gonovote Gauge ───────────────────
ax3 = fig.add_subplot(gs[0, 3])
yes_pct = 68.26
theta  = np.linspace(0, np.pi, 200)
ax3.plot(np.cos(theta), np.sin(theta), 'lightgrey', linewidth=15)
fill_pct = yes_pct / 100
theta_f  = np.linspace(0, fill_pct * np.pi, 200)
ax3.plot(np.cos(theta_f), np.sin(theta_f), '#2ca02c', linewidth=15)
ax3.text(0, 0.1, f"{yes_pct}%", ha='center', va='center', fontsize=16, fontweight='bold', color='#2ca02c')
ax3.text(0, -0.25, "YES in Gonovote", ha='center', fontsize=8, color='grey')
ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-0.5, 1.3)
ax3.axis('off')
ax3.set_title("গণভোট Result", fontweight='bold', fontsize=10)

# ── Panel 4: Gender Bar ───────────────────────
ax4 = fig.add_subplot(gs[1, 0])
genders = ['Male', 'Female', 'Transgender']
values  = [total_male/1e6, total_fem/1e6, total_trans/1000]
colors_g = ['#4472C4', '#ED7D31', '#A9D18E']
bars = ax4.bar(genders, [total_male/1e6, total_fem/1e6, total_trans/1000],
               color=colors_g, edgecolor='white')
ax4.set_ylabel("Count (M / K)")
ax4.set_title("Voter Gender (M=Millions, T=Thousands)", fontweight='bold', fontsize=8)
for bar, v, label in zip(bars, [total_male, total_fem, total_trans],
                          [f'{total_male/1e6:.2f}M', f'{total_fem/1e6:.2f}M', f'{total_trans:,}']):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             label, ha='center', fontsize=8)

# ── Panel 5: Correlation Heatmap (small) ──────
ax5 = fig.add_subplot(gs[1, 1:3])
small_feats = ['Poverty_Rate','Literacy_Rate','Voter_Turnout_Pct','Margin']
small_corr  = df[small_feats].corr(method='pearson')
sns.heatmap(small_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax5, square=True, linewidths=0.5, cbar_kws={'shrink':0.7},
            annot_kws={'size':9})
ax5.set_title("Key Correlation Matrix (Pearson)", fontweight='bold', fontsize=10)

# ── Panel 6: Model Accuracy Comparison ────────
ax6 = fig.add_subplot(gs[1, 3])
model_n  = ['LR','RF','DT']
acc_v    = [results_cls[m]['Accuracy'] for m in results_cls]
f1_v     = [results_cls[m]['F1-Score'] for m in results_cls]
xp = np.arange(3)
ax6.bar(xp - 0.2, acc_v, 0.35, label='Accuracy', color='#4472C4', edgecolor='white')
ax6.bar(xp + 0.2, f1_v,  0.35, label='F1',       color='#ED7D31', edgecolor='white')
ax6.set_xticks(xp)
ax6.set_xticklabels(model_n)
ax6.set_ylim(0, 1.1)
ax6.set_title("ML Model Scores", fontweight='bold', fontsize=10)
ax6.legend(fontsize=7)

# ── Panel 7: Cluster Scatter ──────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
for c in [0, 1, 2]:
    sub = clust_df[clust_df['Cluster'] == c]
    ax7.scatter(sub['Poverty_Rate'], sub['Voter_Turnout_Pct'],
                label=f'Cluster {c}', color=colors_km[c], alpha=0.6, s=25)
ax7.set_xlabel("Poverty Rate (%)")
ax7.set_ylabel("Voter Turnout %")
ax7.set_title("Clusters: Poverty vs Turnout", fontweight='bold', fontsize=10)
ax7.legend(fontsize=8)

# ── Panel 8: Key Stats Table ──────────────────
ax8 = fig.add_subplot(gs[2, 2:4])
ax8.axis('off')
stats_data = [
    ["Total Constituencies",    f"{len(df)}"],
    ["Total Voters",            f"{df['Total_Voters'].sum():,.0f}"],
    ["Avg Voter Turnout",       f"{df['Voter_Turnout_Pct'].mean():.1f}%"],
    ["BNP Seats",               f"{(df['Winner_Party']=='BNP').sum()}"],
    ["Jamaat Seats",            f"{(df['Winner_Party']=='Jamaat').sum()}"],
    ["Gonovote Total",          f"{total_votes/1e6:.2f}M"],
    ["Gonovote YES",            f"{yes_votes/1e6:.2f}M (68.26%)"],
    ["Literacy ↔ Turnout (r)",  f"{pearson_matrix.loc['Literacy_Rate','Voter_Turnout_Pct']:.4f}"],
    ["Poverty ↔ Turnout (r)",   f"{pearson_matrix.loc['Poverty_Rate','Voter_Turnout_Pct']:.4f}"],
    ["Regression R²",           f"{r2:.4f}"],
]
tbl = ax8.table(cellText=stats_data,
                colLabels=['Metric', 'Value'],
                loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#e8f4f8')
ax8.set_title("Key Statistics", fontweight='bold', fontsize=10)

plt.savefig('executive_summary_dashboard.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Executive Summary Dashboard saved.")


# ─────────────────────────────────────────────
# CELL 18 ─ DOWNLOAD ALL OUTPUT FILES
# ─────────────────────────────────────────────
from google.colab import files
import os

output_files = [
    'voter_demographics.png',
    'party_seat_distribution.png',
    'gonovote_analysis.png',
    'voter_turnout_charts.png',
    'correlation_heatmaps.png',
    'geographic_correlation.png',
    'literacy_vs_party.png',
    'poverty_vs_party.png',
    'regression_analysis.png',
    'classification_results.png',
    'kmeans_clustering.png',
    'executive_summary_dashboard.png',
    'bangladesh_election_map.html',
]

print("📥 Downloading all output files...")
for f in output_files:
    if os.path.exists(f):
        files.download(f)
        print(f"  ✅ {f}")
    else:
        print(f"  ⚠️  {f} not found — check if earlier cells ran correctly.")

spark.stop()
print("\n🎉 Analysis complete! All files downloaded.")
