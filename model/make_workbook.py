"""Supporting workbook for the deck (embedded in the PPT as required by the competition rules).
usage: python3 make_workbook.py <sim_out_dir> <open_data_dir> <out.xlsx>"""
import sys, json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter as L
SIM, OD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(f"{SIM}/sim_results.json")); OM = json.load(open(f"{SIM}/open_model_results.json")); ROB = json.load(open(f"{SIM}/robustness.json")); NASA = json.load(open(f"{OD}/nasa_power_climatology.json"))
wb = Workbook(); HF = Font(bold=True, color="FFFFFF"); HFILL = PatternFill("solid", fgColor="502B8B"); TF = Font(bold=True, size=14, color="502B8B"); NOTE = Font(italic=True, color="6B6480"); INP = Font(color="0000FF")
def sheet(name, title, note=None, first=False):
    ws = wb.active if first else wb.create_sheet(); ws.title = name; ws["A1"] = title; ws["A1"].font = TF
    if note: ws["A2"] = note; ws["A2"].font = NOTE
    return ws
def header(ws, row, cols):
    for j, c in enumerate(cols, 1):
        x = ws.cell(row=row, column=j, value=c); x.font = HF; x.fill = HFILL; x.alignment = Alignment(wrap_text=True, vertical="center")
def widths(ws, w):
    for j, v in enumerate(w, 1): ws.column_dimensions[L(j)].width = v

# ---- 1 Read me
ws = sheet("Read me", "The First 100 Hours: supporting workbook", "Asian Paints Alchemy 2026, Case Study 4, Phase I. Team DESTINY, Tata Elxsi.", first=True)
rows = [("Sheet", "What it holds", "Where the numbers come from"),
        ("Simulation", "Forecast error by model input, by read-out window and by spectral noise; coverage of the 90% range", "Team simulation of 120 recipes (sim.py). Simulated, not coating data."),
        ("Robustness", "The simulation re-run with each kinetic setting halved or doubled", "Team simulation (robustness.py)."),
        ("Decisions", "Verdicts at 100 h for 80 held-out simulated recipes, by target life", "Team simulation (open_model.py). Percentages are formulas."),
        ("Real spectra", "Our read-out applied to 320 real ATR-FTIR spectra", "Teramoto, Y. (Kyoto University), Zenodo doi:10.5281/zenodo.22971354, CC BY 4.0; team analysis (open_model.py)."),
        ("India climate", "Monthly UV-A, temperature and humidity for eight Indian cities and Miami; dose ratios", "NASA POWER Climatology API v2.10, retrieved 15 Nov 2026. Annual dose, ratios and humid share are formulas."),
        ("UV dose", "Conversion between chamber hours and South Florida exposure at 340 nm", "ASTM G154 cycle 1 settings and the Atlas South Florida benchmark. All results are formulas; inputs are in blue."),
        ("Pass marks", "Proposed acceptance criteria and gates", "Team proposals, not results."),
        ("Number trace", "Every key number on slides 1-5, its exact value, source and evidence type; exact 95% intervals", "Model outputs and arithmetic; formulas for the intervals."),
        ("Business case", "Annual chamber-time value of the 100-hour gate, from Asian Paints' own inputs", "Formula only: no cost values are assumed.")]
for i, r in enumerate(rows): 
    for j, v in enumerate(r, 1): ws.cell(row=4 + i, column=j, value=v)
header(ws, 4, rows[0]); widths(ws, [16, 78, 90])
import os as _os
LK = [("All submission files (Google Drive folder)", _os.environ.get("LINK_FOLDER")), ("Model code (Word, Google Drive)", _os.environ.get("LINK_CODE")),
      ("Kyoto University open spectra (Zenodo)", "https://doi.org/10.5281/zenodo.22971354"), ("NASA POWER climatology", "https://power.larc.nasa.gov/")]
ws["A16"] = "Links"; ws["A16"].font = Font(bold=True, color="502B8B")
for i, (t, u) in enumerate([x for x in LK if x[1]]):
    c = ws.cell(row=17 + i, column=1, value=t); c.hyperlink = u; c.font = Font(color="502B8B", underline="single")

# ---- 2 Simulation
ws = sheet("Simulation", "Simulation: forecast error using the first 100 h", "SIMULATED DATA, 120 recipes, 5 x 5 repeated cross-validation. Error = median absolute % error of hours to failure.")
header(ws, 4, ["Model input", "Median error, %", "R²"]); r = 5
for k, lab in (("Band features + descriptors", "Our model: infrared bands + recipe facts"), ("Band features (spectra only)", "Infrared bands only"), ("Single carbonyl index", "Single carbonyl index (industry practice)"), ("Formulation descriptors only", "Recipe facts only"), ("Early gloss change only", "Early gloss change")):
    ws.append([lab, round(R["models_100h"][k]["mdape"], 2), round(R["models_100h"][k]["r2"], 3)])
ws.append(["Guessing the average", round(R["mean_baseline_mdape"], 2), None])
ws["A12"] = "Carbonyl index error divided by our model's error"; ws["B12"] = "=B7/B5"
header(ws, 14, ["Read-out window, h", "Our model, %", "Carbonyl index, %", "Early gloss, %"])
for k, v in R["window"].items(): ws.append([int(k), round(v["full"][0], 2), round(v["ci"][0], 2), round(v["gloss"][0], 2)])
n0 = 15 + len(R["window"]) + 1; header(ws, n0, ["Spectral noise, milli-absorbance", "Our model, %", "Carbonyl index, %"])
for k, v in R["noise"].items(): ws.append([float(k) * 1000, round(v["full"][0], 2), round(v["ci"][0], 2)])
n1 = n0 + len(R["noise"]) + 2; header(ws, n1, ["90% range (split conformal)", "Value"])
ws.append(["Share of true values inside the range, %", round(R["conformal"]["coverage"] * 100, 2)]); ws.append(["Range factor (estimate multiplied and divided by)", round(R["conformal"]["factor"], 3)])
widths(ws, [52, 18, 18, 16])

# ---- 3 Robustness
ws = sheet("Robustness", "Simulation robustness: one setting changed at a time", "SIMULATED DATA. Baseline: oxidation start rate 2e-4 per hour (median), stabiliser factor 4, stabiliser fade 400 h, self-acceleration 3.")
header(ws, 4, ["Case", "Our model error, %", "Carbonyl index error, %", "Our model better?", "Median failure time, h"])
for i, o in enumerate(ROB): ws.append([o["case"], round(o["model_error_pct"], 2), round(o["carbonyl_error_pct"], 2), f'=IF(B{5+i}<C{5+i},"yes","no")', round(o["failure_h_p5_p50_p95"][1])])
e = 4 + len(ROB); ws.cell(row=e + 2, column=1, value="Lowest and highest error, our model"); ws.cell(row=e + 2, column=2, value=f"=MIN(B5:B{e})"); ws.cell(row=e + 2, column=3, value=f"=MAX(B5:B{e})")
ws.cell(row=e + 3, column=1, value="Lowest and highest error, carbonyl index"); ws.cell(row=e + 3, column=2, value=f"=MIN(C5:C{e})"); ws.cell(row=e + 3, column=3, value=f"=MAX(C5:C{e})")
ws.cell(row=e + 4, column=1, value="Settings where our model is better"); ws.cell(row=e + 4, column=2, value=f'=COUNTIF(D5:D{e},"yes")'); widths(ws, [40, 20, 24, 18, 22])

# ---- 4 Decisions
ws = sheet("Decisions", "Verdicts at 100 h on 80 held-out simulated recipes", "SIMULATED DATA. Advance if the whole 90% range is above the target life; drop if all of it is below; otherwise run the full test.")
header(ws, 4, ["Target life, h", "Recipes", "Advance", "Drop", "Full test", "Wrong drops", "Wrong advances", "Verdict at 100 h, %", "Chamber hours saved, %"])
ws["K4"] = "Full test length, h"; ws["L4"] = 2000; ws["M4"] = "Read-out length, h"; ws["N4"] = 100; ws["L4"].font = INP; ws["N4"].font = INP
for i, (k, v) in enumerate(OM["decision_analysis"]["by_target_hours"].items()):
    r = 5 + i; ws.append([int(k), v["n"], v["advance"], v["drop"], v["full_test"], v["wrong_drops"], v["wrong_advances"], f"=(C{r}+D{r})/B{r}*100", f"=(C{r}+D{r})*($L$4-$N$4)/(B{r}*$L$4)*100"])
widths(ws, [16, 10, 10, 8, 10, 13, 16, 20, 24, 3, 20, 8, 20, 8])

# ---- 5 Real spectra
ws = sheet("Real spectra", "Our read-out on real ATR-FTIR spectra (Kyoto University open data)", "REAL DATA. Acrylic (semi-film-forming) and alkyd (penetrating) wood coatings under xenon light at 60 and 120 W/m2; 10 spectra per exposure time.")
RS = OM["real_spectra"]; header(ws, 4, ["First 100 h at 60 W/m2", "Coating", "Value at 0 h", "Value at 100 h", "Change, %", "Scatter between repeat spectra, %", "Change as a multiple of the scatter"]); r = 5
for coat, nm in (("SFC", "Acrylic"), ("PC", "Alkyd")):
    for key, lab in (("acid_ester", "Acid / ester band ratio"), ("carbonyl_index", "Carbonyl index (C=O / C-H)")):
        v = RS[f"{coat}_first100h_60W"][key]; ws.append([lab, nm, round(v["at0"], 4), round(v["at100"], 4), f"=(D{r}/C{r}-1)*100", round(v["rel_sd_pct_median"], 2), round(v["signal_to_noise"], 2)]); r += 1
r += 1; header(ws, r, ["Equal UV dose at two lamp intensities", "Coating", "Hours at 120 W/m2", "Value at 120 W/m2", "Value at 60 W/m2 at twice the time", "Difference, %"]); r += 1; r0 = r
for coat, nm in (("SFC", "Acrylic"), ("PC", "Alkyd")):
    v = RS[f"{coat}_dose_collapse"]["acid_ester"]
    for t, a, b in zip(v["times_120W"], v["value_120W"], v["value_60W_at_double_time"]): ws.append(["Acid / ester band ratio", nm, t, a, b, f"=ABS(D{r}/E{r}-1)*100"]); r += 1
ws.cell(row=r + 1, column=1, value="Mean difference, acrylic, %"); ws.cell(row=r + 1, column=2, value=f"=AVERAGE(F{r0}:F{r0+4})"); ws.cell(row=r + 2, column=1, value="Mean difference, alkyd, %"); ws.cell(row=r + 2, column=2, value=f"=AVERAGE(F{r0+5}:F{r0+9})")
r += 4; header(ws, r, ["UV dose read from spectra (train on one intensity, test on the other)", "Coating", "Trained at, W/m2", "Tested at, W/m2", "Six bands, R²", "Carbonyl index, R²", "Test spectra"])
for coat, nm in (("SFC", "Acrylic"), ("PC", "Alkyd")):
    for tr, te in ((60, 120), (120, 60)):
        v = RS[f"{coat}_dose_clock_train{tr}_test{te}"]; ws.append(["Dose in kWh/m2", nm, tr, te, round(v["six_bands"]["r2"], 3), round(v["carbonyl_index"]["r2"], 3), v["six_bands"]["n_test"]])
ws.append([]); ws.append(["Limits: no gloss or failure readings, two coatings, wood not masonry, xenon not ASTM G154, no unexposed controls. This tests the read-out, not the failure-time forecast."]); widths(ws, [58, 12, 18, 18, 30, 30, 30])

# ---- 6 India climate
ws = sheet("India climate", "India climate layer: NASA POWER climatology", "OPEN DATA. UV-A in kWh/m2/day, temperature in C, relative humidity in %. Annual dose, ratios and humid share are formulas.")
MON = NASA["_keys"][:12]; DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; cities = ["Cochin", "Mahabaleshwar", "Mumbai", "Chennai", "Bengaluru", "Kolkata", "Delhi", "Jodhpur", "Miami_Florida"]
ws.append([]); header(ws, 4, ["Days in month"] + MON); ws.append(["days"] + DAYS)
def block(start, title, key):
    ws.cell(row=start, column=1, value=title).font = Font(bold=True, color="502B8B"); header(ws, start + 1, ["City"] + MON)
    for i, c in enumerate(cities):
        for j, v in enumerate([c.replace("_", ", ")] + NASA[c][key][:12], 1): ws.cell(row=start + 2 + i, column=j, value=v)
    return start + 2
u0 = block(7, "UV-A at the surface, kWh/m2/day", "ALLSKY_SFC_UVA"); h0 = block(19, "Relative humidity at 2 m, %", "RH2M"); t0 = block(31, "Air temperature at 2 m, C", "T2M")
ws["A43"] = "Humid month means relative humidity of at least, %"; ws["B43"] = 75; ws["B43"].font = INP; ws["A44"] = "G154 cycle 1 hours per South Florida year (from sheet UV dose)"; ws["B44"] = "='UV dose'!B12"
header(ws, 46, ["City", "UV-A dose, kWh/m2 a year", "Dose vs South Florida", "Share of dose in humid months, %", "Mean temperature, C", "G154 hours a year, UV only"])
fl = u0 + cities.index("Miami_Florida")
for i, c in enumerate(cities):
    r = 47 + i; u = u0 + i; h = h0 + i; t = t0 + i
    ws.cell(row=r, column=1, value=c.replace("_", ", ")); ws.cell(row=r, column=2, value=f"=SUMPRODUCT(B{u}:M{u},$B$5:$M$5)"); ws.cell(row=r, column=3, value=f"=B{r}/$B${47+cities.index('Miami_Florida')}")
    ws.cell(row=r, column=4, value=f"=SUMPRODUCT(B{u}:M{u},$B$5:$M$5,--(B{h}:M{h}>=$B$43))/B{r}*100"); ws.cell(row=r, column=5, value=f"=SUMPRODUCT(B{t}:M{t},$B$5:$M$5)/SUM($B$5:$M$5)"); ws.cell(row=r, column=6, value=f"=C{r}*$B$44")
ws["A57"] = "UV and humidity only. Satellite data for a horizontal surface; panel temperature and time of wetness are not included."; ws["A57"].font = NOTE; widths(ws, [58] + [11] * 12)

# ---- 7 UV dose
ws = sheet("UV dose", "UV dose arithmetic at 340 nm", "ARITHMETIC. Inputs in blue; everything else is a formula. Counts 340 nm UV only: heat and moisture are excluded.")
inp = [("UVA-340 lamp irradiance in ASTM G154 cycle 1, W/m2/nm", 0.89), ("Light hours in each 12-hour cycle", 8), ("South Florida total UV per year (295-385 nm), MJ/m2", 280), ("Share of that UV at 340 nm, per nm", 0.011), ("Xenon arc irradiance at 340 nm, W/m2/nm", 0.55), ("Read-out window, chamber hours", 100), ("Standard lab screen, chamber hours", 2000)]
for i, (a, b) in enumerate(inp): ws.cell(row=4 + i, column=1, value=a); c = ws.cell(row=4 + i, column=2, value=b); c.font = INP
out = [("South Florida dose at 340 nm, kJ/m2/nm a year", "=B6*1000*B7"), ("G154 cycle 1 hours for one South Florida year", "=B11/(B4*3.6*B5/12)"), ("Xenon hours for one South Florida year", "=B11/(B8*3.6)"),
       ("Dose in the read-out window, kJ/m2/nm", "=B4*3.6*B9*B5/12"), ("Read-out window in South Florida days", "=B14/B11*365"), ("Standard lab screen in South Florida years (G154)", "=B10/B12"),
       ("Time saved to a first verdict, %", "=(1-B9/B10)*100"), ("Chamber hours freed per recipe decided at the read-out", "=B10-B9"), ("Read-out window in days", "=B9/24"), ("Standard lab screen in days", "=B10/24")]
for i, (a, b) in enumerate(out): ws.cell(row=11 + i, column=1, value=a); ws.cell(row=11 + i, column=2, value=b)
widths(ws, [62, 16])

# ---- 8 Pass marks
ws = sheet("Pass marks", "Proposed pass marks and gates", "TEAM PROPOSALS, to be frozen before the blind test is opened. Not results.")
header(ws, 4, ["Item", "Pass mark", "Checked at"])
for r in [("Median error of hours to failure on the sealed blind set", "15% or less, and below the carbonyl index", "Gate B, month 11"), ("Rank agreement with the true failure order (Spearman)", "0.80 or more", "Gate B, month 11"), ("Share of true failure times inside the 90% range", "88% to 92%", "Gate B, month 11"),
          ("Recipes refused as unfamiliar", "15% or less", "Gate B, month 11"), ("Dropped recipes that would have passed the full test", "5% or less", "Gate B, month 11"), ("Spectral noise", "1 milli-absorbance or less", "Gate A, month 4"), ("Rank agreement with outdoor ratings at Cochin and Mahabaleshwar", "0.70 or more", "Gate C, year 2")]: ws.append(list(r))
widths(ws, [66, 46, 22])
# ---- 9 Number trace: every key number on slides 1-5, its source, and exact intervals
D2 = OM["decision_analysis"]["by_target_hours"]["2000"]; RS = OM["real_spectra"]; M = R["models_100h"]
ws = sheet("Number trace", "Number trace: every key number on slides 1-5", "Each row: the number as shown, the value behind it, the code output or arithmetic it comes from, and whether it is real data, simulation or arithmetic.")
header(ws, 4, ["Slide", "Number as shown", "Exact value", "What it measures", "Source", "Evidence type"])
f = lambda v, d=2: round(v, d)
TR = [
 (1, "83 days / 4 days", "=2000/24 & 100/24", "Chamber time of the 2,000 h screen and the 100 h read-out at continuous running; 7-day cure excluded", "Sheet UV dose", "Arithmetic"),
 (3, "214 kJ/m2/nm ≈ 25 days", "='UV dose'!B14", "UV dose at 340 nm in 100 h of ASTM G154 cycle 1", "Sheet UV dose", "Arithmetic"),
 (4, "8–11× (acrylic)", f"{f(RS['SFC_first100h_60W']['acid_ester']['signal_to_noise'],1)} / {f(RS['SFC_first100h_60W']['carbonyl_index']['signal_to_noise'],1)}", "Rise 0→100 h ÷ pooled SD of the 0 h and 100 h repeats: acid/ester ratio; carbonyl index", "open_model.py → open_model_results.json", "Real data"),
 (4, "alkyd 5–18×", f"{f(RS['PC_first100h_60W']['acid_ester']['signal_to_noise'],1)} / {f(RS['PC_first100h_60W']['carbonyl_index']['signal_to_noise'],1)}", "Same, alkyd coating", "open_model.py", "Real data"),
 (4, "2–3% (acrylic), alkyd 5%", f"{f(RS['SFC_dose_collapse']['carbonyl_index']['mean_abs_diff_pct'],1)} / {f(RS['SFC_dose_collapse']['acid_ester']['mean_abs_diff_pct'],1)}; alkyd {f(RS['PC_dose_collapse']['acid_ester']['mean_abs_diff_pct'],1)}", "Mean gap between 60 and 120 W/m2 lamps at equal UV dose (alkyd carbonyl index fails: 239%)", "open_model.py", "Real data"),
 (4, "R² 0.72–0.88 / 0.43–0.78", "acrylic, train 60→test 120 and 120→60", "UV dose read back from spectra: six features vs carbonyl index. Not failure time", "open_model.py", "Real data"),
 (4, "alkyd 0.67–0.84 / −0.58–0.64", "alkyd, both directions", "Same, alkyd coating", "open_model.py", "Real data"),
 (4, "11% vs 25%", f"{f(M['Band features + descriptors']['mdape'])} / {f(M['Single carbonyl index']['mdape'])}", "Median absolute % error of hours to failure; 5x5 repeated k-fold CV on 120 simulated recipes; read-out at 100 h", "sim.py → sim_results.json", "Simulation"),
 (4, "10 / 10", "see sheet Robustness", "Settings (each halved or doubled) in which our model's error is below the carbonyl index", "robustness.py", "Simulation"),
 (4, "59% (95% CI 47–70%)", f"{D2['advance'] + D2['drop']} of {D2['n']} = {D2['verdict_at_100h_pct']}%", "Recipes given advance or drop at 100 h, target 2,000 h, of 80 held-out simulated recipes", "open_model.py; sheet Decisions", "Simulation"),
 (4, "0 / 47 (95% bound ≤ 7.5%)", "0 wrong of 47 issued", "Wrong = advanced but truly failed before target, or dropped but truly passed. 33 recipes sent to full test cannot be wrong", "open_model.py", "Simulation"),
 (3, "31% / 41% / 28%", f"{D2['advance']} / {D2['full_test']} / {D2['drop']} of {D2['n']}", "Stage-gate shares at target 2,000 h", "open_model.py", "Simulation"),
 (5, "1,900 h", "=2000-100", "Chamber hours freed per recipe advanced or dropped at 100 h", "Arithmetic", "Arithmetic"),
 (5, "−95%", "=1-100/2000", "Chamber time to a verdict, for recipes that get one", "Arithmetic", "Arithmetic"),
 ("D3", "90% range coverage 91.3%", f"{round(R['conformal']['coverage']*80)} of 80", "True failure times inside the split-conformal 90% range, held-out test set", "sim.py", "Simulation"),
]
for i, row in enumerate(TR):
    for j, v in enumerate(row, 1):
        if j == 3 and isinstance(v, str) and v.startswith("=") and ("&" in v or "D2" in v): v = v[1:]
        ws.cell(row=5 + i, column=j, value=v)
r0 = 5 + len(TR) + 2
ws.cell(row=r0, column=1, value="Exact (Clopper–Pearson) 95% intervals on the simulated test counts").font = Font(bold=True, color="502B8B")
header(ws, r0 + 1, ["Count", "Successes k", "Trials n", "Share", "Lower 95%", "Upper 95%"])
for i, (lab, k, n) in enumerate([("Verdict at 100 h (target 2,000 h)", D2["advance"] + D2["drop"], D2["n"]), ("Wrong verdicts among those issued", D2["wrong_drops"] + D2["wrong_advances"], D2["advance"] + D2["drop"]), ("True value inside the 90% range", round(R["conformal"]["coverage"] * 80), 80)]):
    r = r0 + 2 + i
    ws.cell(row=r, column=1, value=lab); ws.cell(row=r, column=2, value=k); ws.cell(row=r, column=3, value=n); ws.cell(row=r, column=4, value=f"=B{r}/C{r}")
    ws.cell(row=r, column=5, value=f"=IF(B{r}=0,0,_xlfn.BETA.INV(0.025,B{r},C{r}-B{r}+1))"); ws.cell(row=r, column=6, value=f"=IF(B{r}=C{r},1,_xlfn.BETA.INV(0.975,B{r}+1,C{r}-B{r}))")
ws.cell(row=r0 + 6, column=1, value="Intervals describe sampling of the 80-recipe simulated test set only; they do not cover errors in the simulation's own assumptions.").font = NOTE
widths(ws, [8, 28, 26, 70, 36, 14])

# ---- 10 Business case: a formula with Asian Paints' own inputs (no assumed costs)
ws = sheet("Business case", "Business case: annual chamber-time value of a 100-hour gate", "FORMULA ONLY. Blue cells are inputs Asian Paints supplies; they are left empty on purpose (the brief asks for no assumed values). Results appear when the inputs are filled.")
header(ws, 4, ["Input", "Value", "Unit / note"])
INPUTS = [("New exterior recipes entering durability testing per year", None, "recipes / year (Asian Paints)"),
 ("Share of recipes that get an advance or drop verdict at 100 h", None, "fraction 0-1; simulation gives 0.59 at a 2,000 h target (sheet Decisions); replace with the pilot value"),
 ("Standard screen length", 2000, "chamber hours (ASTM G154 screen used in the deck)"),
 ("Read-out window", 100, "chamber hours"),
 ("Panels per recipe in the chamber", None, "panels (deck plan: 3 exposed)"),
 ("Fully loaded cost per panel-slot-hour in the chamber", None, "₹ per panel-slot-hour (lamps, power, maintenance, depreciation, labour)"),
 ("Extra ATR-FTIR cost per recipe", None, "₹ per recipe (≈ 48 spectra: 4 panels × 3 spots × 4 reads; instrument and analyst time)"),
 ("Annual software upkeep, monitoring and retraining", None, "₹ / year"),
 ("One-off build, integration and training", None, "₹ (spread over the years below)"),
 ("Years to spread the one-off cost over", None, "years")]
for i, (a, b, c) in enumerate(INPUTS):
    r = 5 + i; ws.cell(row=r, column=1, value=a); x = ws.cell(row=r, column=2, value=b); x.font = INP; x.fill = PatternFill("solid", fgColor="FFF6E3"); ws.cell(row=r, column=3, value=c)
header(ws, 16, ["Result", "Value", "Formula"])
OUTS = [("Recipes decided at 100 h per year", "=IF(COUNT(B5:B6)=2,B5*B6,\"enter inputs\")", "recipes × share"),
 ("Chamber hours avoided per decided recipe", "=B7-B8", "screen − read-out = 1,900 h"),
 ("Panel-slot-hours avoided per year", "=IF(AND(ISNUMBER(B17),ISNUMBER(B9)),B17*B18*B9,\"enter inputs\")", "decided × hours × panels"),
 ("Gross annual saving, ₹", "=IF(AND(ISNUMBER(B19),ISNUMBER(B10)),B19*B10,\"enter inputs\")", "panel-slot-hours × cost"),
 ("Annual running costs, ₹", "=IF(COUNT(B5,B11,B12,B13,B14)=5,B5*B11+B12+B13/B14,\"enter inputs\")", "ATR-FTIR for every recipe + upkeep + one-off ÷ years"),
 ("Net annual saving, ₹", "=IF(AND(ISNUMBER(B20),ISNUMBER(B21)),B20-B21,\"enter inputs\")", "gross − running"),
 ("Chamber capacity released, % of screen hours", "=IF(ISNUMBER(B6),B6*(B7-B8)/B7*100,\"enter inputs\")", "share decided × 95%")]
for i, (a, b, c) in enumerate(OUTS): r = 17 + i; ws.cell(row=r, column=1, value=a); ws.cell(row=r, column=2, value=b); ws.cell(row=r, column=3, value=c)
ws["A25"] = "The early-decision share is simulated until the pilot measures it; it is not a proven real-world reduction. No savings figure is claimed in the deck."; ws["A25"].font = NOTE
widths(ws, [62, 18, 92])

for w in wb.worksheets: w.sheet_view.showGridLines = True; w.freeze_panes = None
wb.save(OUT); print("saved", OUT, [w.title for w in wb.worksheets])
